import sys
import os
import logging
from io import BytesIO
from threading import Thread, Condition
from queue import Queue
import time
import wave
import traceback
from inspect import signature
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
import logging

import numpy as np
import librosa

import pygame
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, VadOptions, TranscriptionOptions, TranscriptionInfo, decode_audio, format_timestamp, get_speech_timestamps, collect_chunks, Tokenizer, get_suppressed_tokens, restore_speech_timestamps

from voice_utils import RingBuffer, AudioRingBuffer, VadCounter, towave, load_wave

import sounddevice as sd
logger = logging.getLogger('voice')

def _mic_priority(x):
    devid = x['index']
    name = x['name']
    if 'default' in name:
        return 10000 + devid
    if 'USB' in name:
        return 20000 + devid
    return 90000 + devid

def get_mic_devices( *, samplerate=None, dtype=None ):
    """マイクとして使えるデバイスをリストとして返す"""
    # 条件
    sr:float = float(samplerate) if samplerate else 16000
    dtype = dtype if dtype else np.float32
    # select input devices
    inp_dev_list = [ x for x in sd.query_devices() if x['max_input_channels']>0 ]
    # select avaiable devices
    mic_dev_list = []
    for x in inp_dev_list:
        mid = x['index']
        name = f"[{mid:2d}] {x['name']}"
        try:
            # check parameters
            sd.check_input_settings( device=mid, samplerate=sr, dtype=dtype )
            # read audio data
            # channelsを指定したら内臓マイクで録音できないので指定してはいけない。
            with sd.InputStream( samplerate=sr, device=mid ) as audio_in:
                frames,overflow = audio_in.read(1000)
                audio_in.abort(ignore_errors=True)
                audio_in.stop(ignore_errors=True)
                audio_in.close(ignore_errors=True)
                if len(frames.shape)>1:
                    frames = frames[:,0]
                if max(abs(frames))<1e-9:
                    logger.debug(f"NoSignal {name}")
                    continue
            logger.debug(f"Avairable {name}")
            mic_dev_list.append(x)
        except sd.PortAudioError:
            logger.debug(f"NoSupport {name}")
        except:
            logger.exception('mic')
    # sort
    mic_dev_list = sorted( mic_dev_list, key=_mic_priority)
    # for x in mic_dev_list:
    #     print(f"[{x['index']:2d}] {x['name']}")
    return mic_dev_list

audio_queue:Queue = Queue()
xx_callback = None

def audio_callback(data,a,b,c):
    global audio_queue
    mono_f32=data[:,0]
    mono_f32 = mono_f32.reshape(-1,1)
    audio_queue.put( mono_f32 )

def audio_transfer():
    global xx_callback
    print("Thread start")
    while xx_callback is not None:
        try:
            data = audio_queue.get(timeout=1)
            xx_callback(data)
        except:
            pass


def start_mic( *, fr=16000, bs=800, device=None, callback):
    global xx_callback
    if device is None:
        inp_dev_list = get_mic_devices(samplerate=fr, dtype=np.float32)
        device = inp_dev_list[0]['index'] if inp_dev_list and len(inp_dev_list)>0 else None
    xx_callback = callback
    ss = Thread( target=audio_transfer, daemon=True )
    ss.start()
    audioinput = sd.InputStream( samplerate=fr, blocksize=bs, device=device, dtype=np.float32, callback=audio_callback )
    audioinput.start()
    return audioinput

class WhisperSTT:

    def __init__(self):
        self.fr=16000
        self.num_fr = 0
        self.dict_list:list[dict] = []

        self.whisper_model = None
        
        self.vad:VadCounter = VadCounter()
        self.rec:bool=False
        self.last_break_fr = 0
        self.last_transcrib_fr = 0
        self.transcrib_elaps_fr = 0

        self.buffer_start = 0
        self.buffer_fr=0
        self.buffer:AudioRingBuffer = AudioRingBuffer( sec=30, fr=self.fr )
        self.next:int = self.fr*5

    def load(self):
        print( f"#Load model")

        librosa.resample( np.zeros( (1,1), dtype=np.float32), orig_sr=44100, target_sr=16000 ) # preload of librosa

        model_size = "large-v3"
        model_size = "medium"
        # Run on GPU with FP16
        # vself.whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # or run on GPU with INT8
        self.whisper_model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def audio_callback(self,raw_data):
        # マイクからの入力はコピーして使うこと！
        fr1 = self.buffer.get_last_fr()
        self.buffer.append( raw_data[:,0] )
        fr2 = self.buffer.get_last_fr()

        a,b,c,d = self.vad.put_f32( raw_data[:,0] )
        if not a or c or not d:
            self.last_break_fr = fr2

        rec_min_fr = int( 2 * self.fr )
        stop_fr = int( 2 * self.fr )
        rec_split_fr = int( 10 * self.fr)
        keep_startup_fr = int( 1 * self.fr )

        if self.rec:
            # 録音中
            if not a and not b and not d:
                # 無音区間検出
                silent_length = 0
                if self.silent_start_fr<0:
                    self.silent_start_fr = fr2
                    if len(self.buffer)>=rec_min_fr:
                        print( f"[Rec]--silent--split")
                        self.run_transcribe_time()
                    else:
                        print( f"[Rec]--silent--")
                else:
                    silent_length = fr2 - self.silent_start_fr
                    if silent_length>=stop_fr:
                        if self.last_transcrib_fr<self.silent_start_fr:
                            print( f"[Rec]--silent-- cleanup")
                            self.run_transcribe_time()
                        self.rec = False
                        print( f"[Rec]Stop")
            else:
                self.silent_start_fr = -1
                # 録音中に長くなりすぎたら区切る
                if len(self.buffer)>rec_split_fr:
                    print( f"[Rec]Split")
                    self.run_transcribe_time()
        else:
            # 停止中...
            if not a or c or not d:
                print( f"[Rec]start")
                # 音声っぽいのを検出
                clear = len(self.buffer)-keep_startup_fr
                if clear>0:
                    self.buffer.remove(clear)
                self.rec = True
                self.silent_start_fr=-1

    def run_transcribe_time(self):
            t1 = time.time()
            self.last_transcrib_fr = self.buffer.get_last_fr()
            self.run_transcribe()
            t2 = time.time()
            self.transcrib_elaps_fr = int( ((t2-t1+1.0)*self.fr + self.transcrib_elaps_fr ) / 2 )

    def run_transcribe(self):
        print( f"run_transcribe" )
        #wav = self.buffer.towave()
        audio = self.buffer[0:]
        tt0=time.time()
        segments, info = self.whisper_model.transcribe( audio, beam_size=1, best_of=2, temperature=0, language='ja', condition_on_previous_text='まいど！' )
        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        cut_fr = 0
        for segment in segments:
            seg_start_fr = int(segment.start*self.fr)
            fr_start = seg_start_fr + self.buffer.offset
            start_sec = fr_start/self.fr

            seg_end_fr = int(segment.end*self.fr)
            fr_end = seg_end_fr + self.buffer.offset
            end_sec = fr_end/self.fr

            print( f" detect {segment.id} {start_sec:.3f}s-{end_sec:.3f}s {segment.start:.3f}s-{segment.end:.3f}s {seg_start_fr} {seg_end_fr} {segment.text}" )
            cut_fr = seg_start_fr
        tt1 = time.time()
        print( f"  {tt1-tt0}(sec)")
        if cut_fr>0:
            self.buffer.remove(cut_fr)

def main():
    wav_filename='testData/nakagawke01.wav'
   # wav_filename='testData/voice_mosimosi.wav'

    print( f"#Split audio")
    stt:WhisperSTT = WhisperSTT()
    stt.load()

    #load_wave(wav_filename, callback=stt.audio_callback, wait=False )

    audio = start_mic( callback=stt.audio_callback )

    # Pygameの初期化
    pygame.init()
    pygame.mixer.init()
    n=-1
    while True:
        if n<0:
            for idx,item in enumerate(stt.dict_list):
                s = item.get('start',-1)
                e = item.get('end',-1)
                txt = item.get('content','')
                print( f"{idx:3d} {s:8.3f} - {e:8.3f} : {txt}" )
        keyin = input(">> ")
        try:
            n = int(keyin)
        except:
            n=-1
        if 0<=n and n<len(stt.dict_list):
            print(f"{n}")
            au = stt.dict_list[n].get('audio')
            wav = towave( au )
            # オンメモリのwaveデータを読み込む
            wave_sound = pygame.mixer.Sound(wav)
            # 再生
            wave_sound.play()

if __name__ == "__main__":
    main()
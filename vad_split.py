import sys
import os
from io import BytesIO
from threading import Thread, Condition
import time
import wave
import traceback
import numpy as np
import librosa

import pygame
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from voice_utils import RingBuffer, AudioRingBuffer, VadCounter, towave, load_wave

# 細切れにしすぎてまともに音声認識できなかった

class Txx:

    def __init__(self):
        self.vad:VadCounter = VadCounter()
        self.fr=16000
        self.num_fr = 0
        self.dict_list:list[dict] = []
        self.rec = False
        self.whisper_model = None
        self.audio:np.ndarray = None
        self.prev_data = None
        self.buffer_start = 0
        self.buffer_fr=0
        self.buffer:AudioRingBuffer = AudioRingBuffer( sec=30, fr=self.fr )
        self.last_voice_end_fr = 0

    def load(self):
        print( f"#Load model")

        librosa.resample( np.zeros( (1,1), dtype=np.float32), orig_sr=44100, target_sr=16000 ) # preload of librosa

        model_size = "large-v3"
        # Run on GPU with FP16
        # model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # or run on GPU with INT8
        self.whisper_model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def audio_callback(self,raw_data):
        # マイクからの入力はコピーして使うこと！
        mono_data = raw_data[:,0]
        start_time = self.num_fr / self.fr
        end_time = (self.num_fr+len(mono_data))/self.fr
        start_state,up_trigger,dn_trigger,end_state = self.vad.put_f32(mono_data)

        if not self.rec:
            # スキップ中
            if start_state or up_trigger or end_state:
                print( f"Rec Start {start_time:.3f}(sec)")
                self.rec = True
                self.buffer_split = self.num_fr + (self.fr*0.5) # 0.5秒後に音声認識
                self.buffer.clear()
                self.buffer_fr = self.num_fr # バッファの基準位置
                self.last_voice_end_fr = 0
                if self.prev_data is not None:
                    self.buffer_fr = self.num_fr - len(self.prev_data)
                    self.buffer.append( self.prev_data )
                self.buffer.append(mono_data)
        else:
            # 録音中
            self.buffer.append(mono_data)
            if self.num_fr>self.buffer_split:
                self.buffer_split = self.num_fr + (self.fr*0.5) # 0.5秒後に音声認識
                if self.run_transcribe()>1.5:
                    self.rec = False
                    print( f"Rec Stop {start_time:.3f}(sec)")

        self.num_fr += len(raw_data)

    def run_transcribe(self):
        print( f"run_transcribe" )
        wav = self.buffer.towave()
        segments, info = self.whisper_model.transcribe( wav, beam_size=5, language='ja' )
        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        last_fr = 0
        for segment in segments:
            print(" detect %.3f - %.3f %s" % (segment.start, segment.end, segment.text))
            seg_fr = int( segment.end/self.fr)
            if seg_fr>len(self.buffer):
                # なぞ音声
                continue
            st_fr = self.buffer_fr + int( segment.start/self.fr)
            ed_fr = self.buffer_fr + int( segment.end/self.fr)
            last_fr = ed_fr
            self.last_voice_end_fr = ed_fr
        if last_fr>0:
            self.buffer_fr += last_fr
            self.buffer.remove(last_fr) # 認識が終了した部分を削る
        else:
            end_sec:float = len(self.buffer) / self.fr
            if end_sec>0.5:
                sz = int( (end_sec-0.5)*self.fr )
                self.buffer_fr += sz
                self.buffer.remove(sz)
        sirent_sec = (self.num_fr-self.last_voice_end_fr)/self.fr
        return sirent_sec
        faster_whisper.transcribe.segment

def main():
    wav_filename='testData/nakagawke01.wav'

    print( f"#Split audio")
    xxx:Txx = Txx()
    xxx.load()

    load_wave(wav_filename, callback=xxx.audio_callback, wait=False )

    # Pygameの初期化
    pygame.init()
    pygame.mixer.init()
    n=-1
    while True:
        if n<0:
            for idx,item in enumerate(xxx.dict_list):
                s = item.get('start',-1)
                e = item.get('end',-1)
                txt = item.get('content','')
                print( f"{idx:3d} {s:8.3f} - {e:8.3f} : {txt}" )
        keyin = input()
        try:
            n = int(keyin)
        except:
            n=-1
        if 0<=n and n<len(xxx.dict_list):
            print(f"{n}")
            au = xxx.dict_list[n].get('audio')
            wav = towave( au )
            # オンメモリのwaveデータを読み込む
            wave_sound = pygame.mixer.Sound(wav)
            # 再生
            wave_sound.play()

if __name__ == "__main__":
    main()
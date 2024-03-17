import sys
import os
import logging
from io import BytesIO
from threading import Thread, Condition
from queue import Queue
import time
import wave
import traceback
import logging
import math

import numpy as np
import librosa
import matplotlib.pyplot as plt

import pygame
import webrtcvad
import sounddevice as sd
from faster_whisper import WhisperModel

from voice_utils import RingBuffer, AudioRingBuffer,Hists, VadCounter, towave, load_wave, get_mic_devices

logger = logging.getLogger('voice')

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

def rms_energy( audio, sr=16000 ):
    e = librosa.feature.rms( y=audio, hop_length=len(audio))[0][0]
    return e

class VadTbl:

    def __init__(self,size,up:int,dn:int):
        if up<dn:
            raise Exception(f"invalid parameter {size} {up} {dn}")
        self.size = size
        self.up_trigger:int = up
        self.dn_trigger:int = dn
        self.active:bool = False
        self.table:list[int] = [0] * size
        self.pos:int = 0
        self.sum:int = 0

    def add(self,value:int):
        d: int = value - self.table[self.pos]
        self.sum += d
        self.table[self.pos]=value
        self.pos = ( self.pos + 1 ) % self.size
        if self.active:
            if self.sum<=self.dn_trigger:
                self.active = False
                return True
        else:
            if self.sum>=self.up_trigger:
                self.active = True
                return True
        return False


class LowPos:
    def __init__(self, max_elements=10):
        self.max_elements = max_elements
        self.table = np.full((max_elements, 2), np.inf)
        self.current_size = 0
    def __len__(self):
        return self.current_size
    def clear(self):
        np.ndarray.fill( self.table, np.inf )
        self.current_size = 0

    def push(self, pos: int, value: int):
        # 新しい要素を挿入する位置をバイナリサーチで探す
        insert_at = np.searchsorted(self.table[:self.current_size, 0], value)
        
        # 配列がまだ最大要素数に達していない場合
        if self.current_size < self.max_elements:
            # 挿入位置以降の要素を一つ後ろにシフト
            self.table[insert_at+1:self.current_size+1] = self.table[insert_at:self.current_size]
            self.table[insert_at] = [value, pos]
            self.current_size += 1
        else:
            # 配列が満杯で、挿入位置が配列の末尾よりも前の場合
            if insert_at < self.max_elements - 1:
                # 挿入位置以降の要素を一つ後ろにシフトし、末尾の要素を削除
                self.table[insert_at+1:] = self.table[insert_at:-1]
                self.table[insert_at] = [value, pos]

    def get_table(self):
        return self.table[:self.current_size]

    def get_pos(self):
        return int(self.table[0][1])

    def get_posx(self, start, end ):
        for i in range(0,self.current_size):
            pos = int(self.table[i][1])
            if start<=pos and pos<end:
                for j in range(i+1,self.current_size):
                    self.table[j-1] = self.table[j]
                self.current_size-=1
                return pos
        return -1

    def pop(self):
        if self.current_size<=0:
            return None
        ret = int(self.table[0][1])
        for i in range(1,self.current_size):
            self.table[i-1] = self.table[i]
        self.current_size-=1

    def remove_below_pos(self, pos: int):
        # pos以下の要素を削除する
        new_pos = 0
        for i in range(self.current_size):
            if self.table[i][1]>pos:
                self.table[new_pos] = self.table[i]
                new_pos += 1
                
        # 新しい現在のサイズを更新
        self.current_size = new_pos
        
        # 残りの部分をnp.infで埋める
        if new_pos < self.max_elements:
            self.table[new_pos:] = np.inf
    
class VADx:
    """音声の区切りを検出する"""
    def __init__(self, *, callback=None):
        # 設定
        self.fr = 16000
        self.size:int = 10
        self.up_tirg:int = 6
        self.dn_trig:int = 1
        self.min_speech_length = int( 0.7 * self.fr )
        self.max_speech_length = int( 3.5 * self.fr )
        self.max_silent_length = int( 0.7 * self.fr )
        #
        self.callback = callback
        self.dict_list:list[dict] = []
        # frame
        self.frame_msec:int = 10  # 10ms,20ms,30ms
        self.frame_size:int = int( (self.fr * self.frame_msec) / 1000 )  # 10ms,20ms,30ms
        self.frame_buffer_len:int = 0
        self.frame_buffer:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        # 
        self.num_samples = 0
        #
        self.seg_buffer:RingBuffer = RingBuffer( self.fr * 30, dtype=np.float32 )
        self.hists:Hists = Hists( self.seg_buffer.capacity )
        # webrtc-vad
        self.vad = webrtcvad.Vad()
        self.count1:VadTbl = VadTbl( self.size, up=self.up_tirg, dn=self.dn_trig )
        # zero crossing
        self.zc_count:VadTbl = VadTbl( self.size, up=self.up_tirg, dn=self.dn_trig )
        # 判定用 カウンタとフラグ
        self.rec=False
        self.rec_start:int = 0
        self.silent_start:int = 0
        self.seg_dict:dict = None
        # 処理用
        self.last_down:LowPos = LowPos()

    def audio_callback(self, raw_audio:np.ndarray, *args ) ->bool:
        """音声データをself.frame_sizeで分割して処理を呼び出す"""
        try:
            buffer:np.ndarray = self.frame_buffer
            buffer_len:int = self.frame_buffer_len
            mono_f32 = raw_audio[:,0]
            mono_len = len(mono_f32)
            mono_pos = 0
            while mono_pos<mono_len:
                # 分割
                nn = min( mono_len-mono_pos, self.frame_size - buffer_len )
                np.copyto( buffer[buffer_len:buffer_len+nn], mono_f32[mono_pos:mono_pos+nn])
                buffer_len += nn
                mono_pos+=nn
                # framesizeになったら呼び出す
                if buffer_len>=self.frame_size:
                    self._Process_frame( buffer )
                    self.num_samples + buffer_len
                    buffer_len = 0
            self.frame_buffer_len = buffer_len
        except:
            traceback.print_exc()

    def _Process_frame(self, frame:np.ndarray ) ->bool:
        try:
            num_samples = self.num_samples
            # vadカウンタ                
            pcm = frame * 32767.0
            pcm = pcm.astype(np.int16)
            pcm_bytes = pcm.tobytes()
            is_speech = 1 if self.vad.is_speech( pcm_bytes, self.fr ) else 0
            self.count1.add(is_speech)
            # ゼロ交錯数
            zz = librosa.zero_crossings(frame)
            zc = sum(zz)
            self.zc_count.add( zc )
            #
            energy = rms_energy(frame, sr=self.fr )
            #
            self.seg_buffer.append(frame)
            self.hists.add( frame.max(), frame.min(), self.count1.sum, is_speech, energy, zc )

            if self.rec:

                if self.count1.sum<self.count1.size:
                    self.last_down.push( self.seg_buffer.get_pos(), self.count1.sum )

                seg_len = self.seg_buffer.get_pos() - self.rec_start1

                split_len = -1
                if seg_len>=self.min_speech_length:
                    if seg_len>self.max_speech_length:
                        ignore = int( self.fr * 0.5 )
                        split_len = self.last_down.get_posx( self.rec_start1 + ignore, self.seg_buffer.get_pos() - ignore )
                    elif not self.count1.active:
                        split_len = self.seg_buffer.get_pos()

                if split_len>0:

                    self.seg_dict['end'] = split_len
                    self.seg_dict['end_sec'] = split_len/self.fr

                    b = self.seg_buffer.to_index( self.rec_start1 )
                    e = self.seg_buffer.to_index( split_len )
                    audio = self.seg_buffer.to_numpy( b, e )
                    self.seg_dict['audio'] = audio

                    b = self.hists.to_index( self.rec_start1 // self.frame_size )
                    e = self.hists.to_index( split_len//self.frame_size )
                    hist = self.hists.to_numpy( b, e )
                    self.seg_dict['hists'] = hist

                    if self.callback is None:
                        self.dict_list.append( self.seg_dict )
                    else:
                        self.callback(self.seg_dict)

                    if not self.count1.active:
                        self.seg_dict = None
                        self.rec=False
                        self.silent_start = split_len
                    else:
                        self.rec_start1 = split_len
                        print( f"split {self.rec_start1/self.fr}" )
                        self.seg_dict = { 'start': self.rec_start1, 'start_sec': self.rec_start1/self.fr, 'split':1 }
                        self.last_down.remove_below_pos(split_len)
            else:
                if self.count1.active:
                    lenx = len(self.hists)
                    sz = 0
                    while (sz+1)<lenx:
                        v1 = self.hists.get_vad_count( lenx - 1 - sz )
                        v2 = self.hists.get_vad_count( lenx - 1 - sz-1 )
                        if v2==0 or v2>v1:
                            break
                        sz+=1

                    self.rec=True
                    self.rec_start1 = max( 0, self.seg_buffer.get_pos() - (self.frame_size * sz) )
                    self.last_down.clear()
                    self.seg_dict = { 'start': self.rec_start1, 'start_sec': self.rec_start1/self.fr }
                else:
                    if self.silent_start>0 and (self.seg_buffer.get_pos() - self.silent_start)>self.max_silent_length:
                        seg_dict = { 'start': self.silent_start, 'start_sec': self.silent_start/self.fr, 'end': self.silent_start, 'end_sec': self.silent_start/self.fr }
                        self.silent_start = 0
                        if self.callback is None:
                            self.dict_list.append( seg_dict )
                        else:
                            self.callback(seg_dict)
        except:
            traceback.print_exc()

    def end(self):
        if self.rec:
            split_len = self.seg_buffer.get_pos()
            self.seg_dict['end'] = split_len
            self.seg_dict['end_sec'] = split_len/self.fr

            b = self.seg_buffer.to_index( self.rec_start1 )
            e = self.seg_buffer.to_index( split_len )
            audio = self.seg_buffer.to_numpy( b, e )
            self.seg_dict['audio'] = audio

            b = self.hists.to_index( self.rec_start1 // self.frame_size )
            e = self.hists.to_index( split_len//self.frame_size )
            hist = self.hists.to_numpy( b, e )
            self.seg_dict['hists'] = hist

            if self.callback is None:
                self.dict_list.append( self.seg_dict )
            else:
                self.callback(self.seg_dict )

class WhisperSTT:

    def __init__(self):
        self.fr=16000
        self.num_fr = 0
        self.dict_list:list[dict] = []
       
        self.vad:VadCounter = VadCounter()
        self.vad_buffer:RingBuffer = RingBuffer( self.vad.size, dtype=np.float32 )
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
        # self.whisper_model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
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

import matplotlib.pyplot as plt
import librosa
import librosa.display

def analyze_audio2(audio_data, sr):
    # Figureと複数のAxesを作成
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    # 波形のプロット
    librosa.display.waveshow(audio_data, sr=sr, ax=axs[0],color='blue')
    axs[0].set_title('Waveform')

    # RMSエナジー
    rms_energy = librosa.feature.rms(y=audio_data)[0]
    axs[1].semilogy(rms_energy, label='RMS Energy')
    axs[1].legend()
    axs[1].set_title('RMS Energy')

    # ゼロ交差数
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)[0]
    axs[2].plot(zero_crossings, label='Zero Crossing Rate')
    axs[2].legend()
    axs[2].set_title('Zero Crossing Rate')

    # 基本周波数（F0）
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    axs[3].plot(times, f0, label='F0', color='green')
    axs[3].legend()
    axs[3].set_title('Fundamental Frequency (F0)')

    # 各Axesの設定
    for ax in axs:
        ax.label_outer()

    plt.tight_layout()

    # Figureオブジェクトを返す
    return fig

def analyze_audio(audio_data, sr):
    # Figureと複数のAxesを作成
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    # 波形のプロット
    librosa.display.waveshow(audio_data, sr=sr, ax=axs[0], color='blue')
    axs[0].set_title('Waveform')

    # RMSエナジー
    rms_energy = librosa.feature.rms(y=audio_data)[0]
    frames = np.arange(len(rms_energy))
    t_rms = librosa.frames_to_time(frames, sr=sr)
    axs[1].semilogy(t_rms, rms_energy, label='RMS Energy', color='orange')
    axs[1].legend()
    axs[1].set_title('RMS Energy')

    # ゼロ交差数
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)[0]
    frames = np.arange(len(zero_crossings))
    t_zcr = librosa.frames_to_time(frames, sr=sr)
    axs[2].plot(t_zcr, zero_crossings, label='Zero Crossing Rate', color='purple')
    axs[2].legend()
    axs[2].set_title('Zero Crossing Rate')

    # 基本周波数（F0）
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    t_f0 = librosa.times_like(f0, sr=sr)
    axs[3].plot(t_f0, f0, label='F0', color='green')
    axs[3].legend()
    axs[3].set_title('Fundamental Frequency (F0)')

    # 各Axesの設定
    for ax in axs:
        ax.label_outer()

    plt.tight_layout()

    # Figureオブジェクトを返す
    return fig

def plot_hists_data(hists, sr, framesize):
    # Figureと5つのAxesを作成
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)

    # サンプルレートに基づいて時間軸を計算
    frames = np.arange(hists.shape[1]) * framesize
    t = frames / sr

    # 最高値と最低値のプロット
    hi = hists[0]
    lo = hists[1]
    axs[0].plot(t, hi, label='Highest Value', color='blue')
    axs[0].plot(t, lo, label='Lowest Value', color='blue')
    #axs[0].legend()
    axs[0].set_title('Highest and Lowest Values')

    # webrtcvadの結果1と結果2のプロット
    vad_count = hists[2]
    vad = hists[3]
    axs[1].plot(t, vad_count, label='VAD count', color='orange')
    axs[1].plot(t, vad, label='VAD', color='purple')
    axs[1].legend()
    axs[1].set_title('WebRTC VAD Results')

    # RMSエナジー
    energy = hists[4]
    axs[2].plot(t, energy, label='RMS Energy', color='green')
    #axs[2].legend()
    axs[2].set_title('RMS Energy')

    # ゼロ交差数のプロット
    zc = hists[5]
    axs[3].plot(t, zc, label='Zero Crossing Rate', color='green')
    #axs[3].legend()
    axs[3].set_title('Zero Crossing Rate')

    # 各Axesの設定
    for ax in axs[:-1]:
        ax.label_outer()
    axs[-1].set_xlabel('Time (s)')

    plt.tight_layout()

    # Figureオブジェクトを返す
    return fig

def main():

    model_size = "large-v3"
    whisper_model = None

    # wav_filename='testData/nakagawke01.wav'
    wav_filename='testData/voice_mosimosi.wav'
    # wav_filename='testData/voice_command.wav'

    print( f"#Split audio")
    #stt:WhisperSTT = WhisperSTT()
    #stt.load()

    stt_dict_list:list[dict] = []
    def seg_callback( item ):
        stt_dict_list.append(item)
    stt = VADx( callback=seg_callback )
    load_wave( wav_filename, callback=stt.audio_callback, wait=False )
    stt.end()
    # audio = start_mic( callback=stt.audio_callback )

    # Pygameの初期化
    pygame.init()
    pygame.mixer.init()
    n=-1
    selected=-1
    while True:
        if n<0:
            for idx,item in enumerate(stt_dict_list):
                mark = "*" if idx==selected else " "
                split = "|" if item.get('split') is not None else " "
                s = item.get('start_sec',-1)
                e = item.get('end_sec',-1)
                txt = item.get('content','')
                print( f"{mark} {idx:3d} {split} {s:8.3f} - {e:8.3f} : {txt}" )
        mark = f"[{selected:3d}]" if selected>=0 else "[---]"
        keyin = input(f"[{mark} >> ")
        if keyin=="t" and 0<=selected and selected<len(stt_dict_list):
            item = stt_dict_list[selected]
            text = item.get('content')
            if text is None:
                audio = item.get('audio')
                if whisper_model is None:
                    print( f"loading {model_size}")
                    whisper_model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                segments, info = whisper_model.transcribe( audio, beam_size=1, best_of=2, temperature=0, language='ja', condition_on_previous_text='まいど！' )
                text = ""
                for segment in segments:
                    text = text + "//" + segment.text
                print(f"content:{text}")
                item['content'] = text
        try:
            n = int(keyin)
        except:
            n=-1
        if 0<=n and n<len(stt_dict_list):
            item = stt_dict_list[n]
            au = item.get('audio')
            if selected != n:
                selected = n
                try:
                    plt.close('all')
                except:
                    pass
                fig = item.get('fig')
                if fig is None:
                    hists = item['hists']
                    fig = item['fig'] = plot_hists_data( hists, sr=16000, framesize=stt.frame_size )
                    #fig = item['fig'] = analyze_audio(au, sr=16000)
                for p in range(10):
                    try:
                        fig.show()
                        break
                    except:
                        # traceback.print_exc()
                        time.sleep(.5)

            wav = towave( au )
            # オンメモリのwaveデータを読み込む
            wave_sound = pygame.mixer.Sound(wav)
            # 再生
            wave_sound.play(fade_ms=0)

def tbl_test():
    tbl:VadTbl = VadTbl(5, up=4, dn=1 )
    
    hist:list = []
    sgn=1
    for ai in range(1,60,3):
        i = ai * sgn
        sgn = 1 if sgn<0 else -1
        hist.append(i)
        tbl.add(i)
        s = sum(hist[-tbl.size:])
        print( f"sum:{tbl.sum} {s}")

if __name__ == "__main__":
    # tbl_test()
    main()
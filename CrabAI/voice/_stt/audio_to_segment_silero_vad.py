import os
import time
from logging import getLogger
import numpy as np
import pandas as pd
from threading import Thread, Condition
import wave
import webrtcvad
import librosa
from scipy import signal
from .stt_data import SttData
from .ring_buffer import RingBuffer
from .hists import Hists
from .low_pos import LowPos
from ..voice_utils import voice_per_audio_rate
from .silero_vad import SileroVAD

logger = getLogger('audio_to_segment')

def rms_energy( audio, sr=16000 ):
    e = librosa.feature.rms( y=audio, hop_length=len(audio))[0][0]
    return e

class AudioToSegmentSileroVAD:
    """音声の区切りを検出する"""
    def __init__(self, *, callback, sample_rate:int=16000, wave_dir=None):
        # 設定
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000
        self.pick_tirg:float = 0.3
        self.up_tirg:float = 0.45
        self.dn_trig:float = 0.2
        self.ignore_length:int = int( 0.1 * self.sample_rate )
        self.min_speech_length:int = int( 0.8 * self.sample_rate )
        self.max_speech_length:int = int( 4.8 * self.sample_rate )
        self.max_silent_length:int = int( 0.8 * self.sample_rate )
        self.prefech_length:int = int( 1.6 * self.sample_rate )
        #
        self._mute:bool = False
        self.callback = callback
        self.dict_list:list[dict] = []
        # フレーム分割用のバッファ
        self.frame_msec:int = 10  # 10ms,20ms,30ms
        self.frame_size:int = 512 # int( (self.sample_rate * self.frame_msec) / 1000 )  # 10ms,20ms,30ms
        self.frame_buffer_len:int = 0
        self.frame_buffer_raw:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        self.frame_buffer_cut:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        # 
        self.num_samples:int = 0
        #
        self.seg_buffer:RingBuffer = RingBuffer( self.sample_rate * 30, dtype=np.float32 )
        self.raw_buffer:RingBuffer = RingBuffer( self.seg_buffer.capacity, dtype=np.float32 )
        self.hists:Hists = Hists( self.seg_buffer.capacity )
        # webrtc-vad
        # SileroVAD
        self.silerovad:SileroVAD = SileroVAD( window_size_samples=self.frame_size, sampling_rate=self.sample_rate )
        #
        self.var1 = 0.3
        # 判定用 カウンタとフラグ
        self.rec:int = 0
        self.rec_start:int = 0
        self.silent_start:int = 0
        self.stt_data:SttData = None
        self.ignore_list:RingBuffer = RingBuffer( 10, dtype=np.int64)
        # 処理用
        self.last_down:LowPos = LowPos()
        # プリフェッチ用フラグ
        self.prefed:bool=False
        # wave保存用
        self.wave_dir:str = wave_dir
        if wave_dir is not None:
            if not os.path.exists(wave_dir):
                logger.info(f"crete {wave_dir}")
                os.makedirs(wave_dir)
            elif not os.path.isdir(wave_dir):
                raise IOError(f"is not directory {wave_dir}")
        self.wave_lock:Condition = Condition()
        self.wave_stream = None
        self.save_1:int = -1
        self.save_2:int = -1
        self.last_dump:int = 0
        # 人の声のフィルタリング（バンドパスフィルタ）
        # self.sos = scipy.signal.butter( 4, [100, 2000], 'bandpass', fs=self.sample_rate, output='sos')
        # ハイパスフィルタ
        fpass = 60
        fstop = 5
        gpass = 3
        gstop = 60
        fn = self.sample_rate / 2   #ナイキスト周波数
        wp = fpass / fn  #ナイキスト周波数で通過域端周波数を正規化
        ws = fstop / fn  #ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
        self.b, self.a = signal.butter(N, Wn, "high")   #フィルタ伝達関数の分子と分母を計算

    def hipass(self,x):
        y = signal.filtfilt(self.b, self.a, x)    #信号に対してフィルタをかける
        return y  

    def __getitem__(self,key):
        if 'vad.pick'==key:
            return self.pick_tirg
        elif 'vad.up'==key:
            return self.up_trig
        elif 'vad.dn'==key:
            return self.dn_trig
        elif 'var1'==key:
            return self.var1
        return None

    def to_dict(self)->dict:
        keys = ['vad.pick','vad.up','vad.dn','var1']
        ret = {}
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        if 'vad.pick'==key:
            if isinstance(val,(int,float)) and 0<=key<=1:
                self.pick_tirg = float(key)
        elif 'vad.up'==key:
            if isinstance(val,(int,float)) and 0<=key<=1:
                self.up_trig = float(key)
        elif 'vad.dn'==key:
            if isinstance(val,(int,float)) and 0<=key<=1:
                self.dn_trig = float(key)
        elif 'var1'==key:
            if isinstance(val,(int,float)) and 0<=key<=1:
                self.var1 = float(key)

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def load(self):
        try:
            self.silerovad.load()
        except:
            logger.exception("")

    def start(self):
        self.frame_buffer_len=0
        self.num_samples = 0
        self.last_dump = 0

    def set_pause(self,b):
        self._mute = b

    def stop(self):
        pass

    def audio_callback(self, utc:float, raw_audio:np.ndarray, *args ) ->bool:
        """音声データをself.frame_sizeで分割して処理を呼び出す"""
        try:
            if raw_audio is None:
                # End of stream
                self._Process_frame( utc, None,None )
                self.frame_buffer_len = 0
                return

            buffer_raw:np.ndarray = self.frame_buffer_raw
            buffer_cut:np.ndarray = self.frame_buffer_cut
            buffer_len:int = self.frame_buffer_len
            mono_f32 = raw_audio[:,0]
            mono_cut = self.hipass(mono_f32) # ローカットフィルタ

            mono_len = len(mono_f32)
            mono_pos = 0
            while mono_pos<mono_len:
                # 分割
                nn = min( mono_len-mono_pos, self.frame_size - buffer_len )
                np.copyto( buffer_raw[buffer_len:buffer_len+nn], mono_f32[mono_pos:mono_pos+nn])
                np.copyto( buffer_cut[buffer_len:buffer_len+nn], mono_cut[mono_pos:mono_pos+nn])
                buffer_len += nn
                mono_pos+=nn
                # framesizeになったら呼び出す
                if buffer_len>=self.frame_size:
                    self._Process_frame( utc, buffer_raw, buffer_cut )
                    self.num_samples += buffer_len
                    buffer_len = 0
            self.frame_buffer_len = buffer_len
        except:
            logger.exception(f"")

    def _Process_frame(self, utc:float, frame_raw:np.ndarray, frame:np.ndarray ) ->bool:
        try:
            if frame is None:
                return

            num_samples = self.num_samples
            # vadカウンタ                
            is_speech:float = self.silerovad.is_speech( frame )
            # ゼロ交錯数
            zz = librosa.zero_crossings(frame)
            zc = sum(zz)
            #
            energy = rms_energy(frame, sr=self.sample_rate )
            #
            self.seg_buffer.append(frame)
            self.raw_buffer.append(frame_raw)
            self.hists.add( frame.max(), frame.min(), self.rec, is_speech, is_speech, energy, zc, 0.0 )

            if self._mute:
                self.rec=0
                self.rec_start = 0
                self.stt_data = None

            if self.rec>=2:

                if is_speech<self.up_tirg:
                    self.last_down.push( self.seg_buffer.get_pos(), is_speech )

                seg_len = self.seg_buffer.get_pos() - self.rec_start

                end_pos = -1
                if seg_len>self.max_speech_length:
                    ignore = int( self.sample_rate * 0.3 )
                    end_pos = self.last_down.get_posx( self.rec_start + ignore, self.seg_buffer.get_pos() - ignore )
                    if end_pos<0 and not self.prefed:
                        # プリフェッチを出す
                        self.prefed = True
                        self.stt_data.typ = SttData.PreSegment
                        end_pos = self.rec_start + self.prefech_length
                        logger.debug(f"rec prefech {is_speech} {self.rec_start/self.sample_rate}")

                elif seg_len>=self.min_speech_length and is_speech<self.dn_trig:
                    end_pos = self.seg_buffer.get_pos()

                if end_pos>0:
                    self._flush( self.stt_data, end_pos)

                    if is_speech<self.dn_trig:
                        # print(f"rec stop {is_speech} {(split_len-self.rec_start)/self.sample_rate}")
                        self.stt_data = None
                        self.rec=0
                        self.silent_start = end_pos

                    elif self.stt_data.typ==SttData.PreSegment:
                        # 作り直し
                        self.stt_data = SttData( SttData.Segment, utc, self.stt_data.start, 0, self.sample_rate )

                    else:
                        self.rec_start = end_pos
                        # print(f"rec split {is_speech} {self.rec_start/self.sample_rate}")
                        logger.debug(f"rec split {is_speech} {self.rec_start/self.sample_rate}")
                        self.stt_data = SttData( SttData.Segment, utc, self.rec_start, 0, self.sample_rate )
                        self.last_down.remove_below_pos(end_pos)
                        self.prefed = False

            elif self.rec == 1:
                # 音声の先頭部分
                seg_len = self.seg_buffer.get_pos() - self.rec_start
                if is_speech>=self.up_tirg:
                    if seg_len>self.ignore_length:
                        # 音声が規定の長さを超た
                        tmpbuf = self.seg_buffer.to_numpy( -seg_len )
                        var = voice_per_audio_rate(tmpbuf, sampling_rate=self.sample_rate)
                        self.hists.replace_var( var )
                        if var>self.var1:
                            # FFTでも人の声っぽいのでセグメントとして認定
                            print( f"segment start voice/audio {var}" )
                            self.rec = 2
                            # print(f"rec start {self.seg_buffer.get_pos()} {is_speech}")
                            # 直全のパルスをマージする
                            base = self.rec_start
                            x1 = int(self.sample_rate*2.0)
                            x2 = int(self.sample_rate*4.0)
                            i=len(self.ignore_list)-1
                            while i>=0:
                                if base-self.ignore_list[i]>x2:
                                    break
                                if self.rec_start-self.ignore_list[i]<=x1:
                                    self.rec_start = self.ignore_list[i]
                                i-=1
                            self.ignore_list.clear()
                            # 上り勾配をマージする
                            hx = (base-self.rec_start)//self.frame_size
                            lenx = len(self.hists)
                            sz = 0
                            while (hx+sz+1)<lenx:
                                v1 = self.hists.get_vad_count( lenx - 1 -hx - sz )
                                if v1<self.pick_tirg:
                                    break
                                sz+=1
                            self.rec_start = max( 0, self.rec_start - (self.frame_size * sz ))
                            self.last_down.clear()
                            self.prefed = False
                            self.stt_data = SttData( SttData.Segment, utc, self.rec_start, 0, self.sample_rate )
                        else:
                            print( f"ignore pulse voice/audio {var}" )
                else:
                    # 規定時間より短く音声が終了
                    self.rec = 0
                    print(f"rec pulse {self.seg_buffer.get_pos()} {seg_len/self.sample_rate:.3f}")
                    self.ignore_list.add(self.rec_start)
                    self.rec_start = 0
            else:
                # 音声未検出状態
                if is_speech>=self.up_tirg and not self._mute:
                    # 音声を検出した
                    self.rec=1
                    self.rec_start = self.seg_buffer.get_pos()
                    print(f"rec up {self.seg_buffer.get_pos()} {is_speech}")

                else:
                    # 音声じゃない
                    # 無音通知処理
                    if self.silent_start>0 and (self.seg_buffer.get_pos() - self.silent_start)>self.max_silent_length:
                        stt_data = SttData( SttData.Term, utc, self.silent_start, self.silent_start, self.sample_rate )
                        self.silent_start = 0
                        if self.callback is None:
                            self.dict_list.append( stt_data )
                        else:
                            self.callback(stt_data)
            self.hists.replace_color( self.rec )
            if (num_samples-self.last_dump)+len(frame)*2>self.seg_buffer.capacity:
                self.last_dump = num_samples+len(frame)
                ed = self.seg_buffer.get_pos()
                st = max(0, ed - self.seg_buffer.capacity)
                dmp:SttData = SttData( SttData.Dump, utc, st, ed, self.sample_rate )
                self._flush( dmp, ed )
        except:
            logger.exception(f"")

    def _flush(self,stt_data:SttData,end_pos):
            start_pos = stt_data.start
            stt_data.end = end_pos

            b = self.seg_buffer.to_index( start_pos )
            e = self.seg_buffer.to_index( end_pos )
            audio = self.seg_buffer.to_numpy( b, e )
            stt_data.audio = audio
            raw = self.raw_buffer.to_numpy( b, e )
            stt_data.raw = raw

            b = self.hists.to_index( start_pos // self.frame_size )
            e = self.hists.to_index( end_pos//self.frame_size )
            hists = self.hists.to_df( b, e )
            stt_data.hists = hists

            if self.callback is None:
                self.dict_list.append( stt_data )
            else:
                self.callback(stt_data)
    
    def end(self):
        if self.rec:
            end_pos = self.seg_buffer.get_pos()
            self._flush(self.stt_data,end_pos)

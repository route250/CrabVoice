import os
import time
from logging import getLogger
import numpy as np
import pandas as pd
from threading import Thread, Condition
import wave
import webrtcvad
import librosa
#import scipy
from .stt_data import SttData
from .ring_buffer import RingBuffer
from .hists import Hists
from .vad_counter import VadTbl
from .low_pos import LowPos
from ..voice_utils import voice_per_audio_rate

logger = getLogger('audio_to_segment')

def rms_energy( audio, sr=16000 ):
    e = librosa.feature.rms( y=audio, hop_length=len(audio))[0][0]
    return e

class AudioToSegmentWebrtcVAD:
    """音声の区切りを検出する"""
    def __init__(self, *, callback, sample_rate:int=16000, wave_dir=None):
        # 設定
        self.sample_rate = sample_rate if isinstance(sample_rate,int) else 16000
        self.size:int = 20
        self.up_tirg:int = 18
        self.dn_trig:int = 0
        self.ignore_length:int = int( 0.1 * self.sample_rate )
        self.min_speech_length:int = int( 0.8 * self.sample_rate )
        self.max_speech_length:int = int( 4.8 * self.sample_rate )
        self.max_silent_length:int = int( 0.8 * self.sample_rate )
        self.prefech_length:int = int( 1.6 * self.sample_rate )
        #
        self._mute:bool = False
        self.callback = callback
        self.dict_list:list[dict] = []
        # frame
        self.frame_msec:int = 10  # 10ms,20ms,30ms
        self.frame_size:int = int( (self.sample_rate * self.frame_msec) / 1000 )  # 10ms,20ms,30ms
        self.frame_buffer_len:int = 0
        self.frame_buffer:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        # 
        self.num_samples = 0
        #
        self.seg_buffer:RingBuffer = RingBuffer( self.sample_rate * 30, dtype=np.float32 )
        self.hists:Hists = Hists( self.seg_buffer.capacity )
        # webrtc-vad
        self.vad_mode = 2
        self.vad = webrtcvad.Vad(self.vad_mode) # 0:甘い 0:厳しい
        self.count1:VadTbl = VadTbl( self.size, up=self.up_tirg, dn=self.dn_trig )
        #
        self.var1 = 0.3
        # zero crossing
        self.zc_count:VadTbl = VadTbl( self.size, up=self.up_tirg, dn=self.dn_trig )
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

    def __getitem__(self,key):
        if 'vad.mode'==key:
            return self.vad_mode
        elif 'vad.up'==key:
            return self.up_trig
        elif 'vad.dn'==key:
            return self.dn_trig
        elif 'var1'==key:
            return self.var1
        return None

    def to_dict(self)->dict:
        keys = ['vad.mode','vad.up','vad.dn','var1']
        ret = {}
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        if 'vad.mode'==key:
            if isinstance(val,(int,float)) and 0<=key<=3:
                self.vad_mode = int(key)
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
        pass

    def start(self):
        self.frame_buffer_len=0
        self.num_samples = 0
        self.last_dump = 0

    def set_pause(self,b):
        self._mute = b

    def stop(self):
        pass

    def audio_callback(self, utc, raw_audio:np.ndarray, *args ) ->bool:
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
                    self._Process_frame( utc, buffer, buffer )
                    self.num_samples += buffer_len
                    buffer_len = 0
            self.frame_buffer_len = buffer_len
        except:
            logger.exception(f"")

    def _Process_frame(self, utc:float, frame_raw:np.ndarray, frame:np.ndarray ) ->bool:
        try:
            num_samples = self.num_samples
            # vadカウンタ                
            pcm = frame * 32767.0
            pcm = pcm.astype(np.int16)
            pcm_bytes = pcm.tobytes()
            is_speech = 1 if self.vad.is_speech( pcm_bytes, self.sample_rate ) else 0
            self.count1.add(is_speech)
            # ゼロ交錯数
            zz = librosa.zero_crossings(frame)
            zc = sum(zz)
            self.zc_count.add( zc )
            #
            energy = rms_energy(frame, sr=self.sample_rate )
            #
            self.seg_buffer.append(frame)
            self.hists.add( frame.max(), frame.min(), self.rec, self.count1.sum, energy, zc, 0.0 )

            if self._mute:
                self.rec=0
                self.rec_start = 0
                self.stt_data = None

            if self.rec>=2:

                if self.count1.sum<self.count1.size:
                    self.last_down.push( self.seg_buffer.get_pos(), self.count1.sum )

                seg_len = self.seg_buffer.get_pos() - self.rec_start

                split_len = -1
                if seg_len>self.max_speech_length:
                    ignore = int( self.sample_rate * 0.3 )
                    split_len = self.last_down.get_posx( self.rec_start + ignore, self.seg_buffer.get_pos() - ignore )
                    if split_len<0 and not self.prefed:
                        # プリフェッチを出す
                        self.prefed = True
                        self.stt_data.typ = SttData.PreSegment
                        split_len = self.rec_start + self.prefech_length
                        logger.debug(f"rec prefech {self.count1.sum} {self.rec_start/self.sample_rate}")

                elif seg_len>=self.min_speech_length and not self.count1:
                    split_len = self.seg_buffer.get_pos()

                if split_len>0:
                    self._flush(split_len)

                    if not self.count1.active:
                        # print(f"rec stop {self.count1.sum} {(split_len-self.rec_start)/self.sample_rate}")
                        self.stt_data = None
                        self.rec=0
                        self.silent_start = split_len
                        self._wave_end()

                    elif self.stt_data.typ==SttData.PreSegment:
                        # 作り直し
                        self.stt_data = SttData( SttData.Segment, utc, self.stt_data.start, 0, self.sample_rate )

                    else:
                        self.rec_start = split_len
                        # print(f"rec split {self.count1.sum} {self.rec_start/self.sample_rate}")
                        logger.debug(f"rec split {self.count1.sum} {self.rec_start/self.sample_rate}")
                        self.stt_data = SttData( SttData.Segment, utc, self.rec_start, 0, self.sample_rate )
                        self.last_down.remove_below_pos(split_len)
                        self.prefed = False

            elif self.rec == 1:
                seg_len = self.seg_buffer.get_pos() - self.rec_start
                if self.count1:
                    if seg_len>self.ignore_length:
                        tmpbuf = self.seg_buffer.to_numpy( -seg_len )
                        var = voice_per_audio_rate(tmpbuf, sampling_rate=self.sample_rate)
                        self.hists.replace_var(var)
                        if var>self.var1:
                            print( f"segment start voice/audio {var}" )
                            self.rec = 2
                            self._wave_start()
                            # print(f"rec start {self.seg_buffer.get_pos()} {self.count1.sum}")
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
                            lenx = len(self.hists)
                            sz = 0
                            while (sz+1)<lenx:
                                v1 = self.hists.get_vad_count( lenx - 1 - sz )
                                v2 = self.hists.get_vad_count( lenx - 1 - sz-1 )
                                if v2==0 or v2>v1:
                                    break
                                sz+=1
                            self.rec_start = max( 0, self.rec_start - (self.frame_size * sz ))
                            self.last_down.clear()
                            self.prefed = False
                            self.stt_data = SttData( SttData.Segment, utc, self.rec_start, 0, self.sample_rate )
                        # else:
                        #     print( f"ignore pulse voice/audio {var}" )
                else:
                    self.rec = 0
                    # print(f"rec pulse {self.seg_buffer.get_pos()} {seg_len/self.sample_rate:.3f}")
                    self.ignore_list.add(self.rec_start)
                    self.rec_start = 0
            else:
                if self.count1 and not self._mute:
                    self.rec=1
                    self.rec_start = self.seg_buffer.get_pos()
                    # print(f"rec up {self.seg_buffer.get_pos()} {self.count1.sum}")

                else:
                    self._wave_close()
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
                st = ed - self.seg_buffer.capacity
                dmp:SttData = SttData( SttData.Dump, utc, st, ed, self.sample_rate )
                self._flush( dmp, ed )
        except:
            logger.exception(f"")

    def _flush(self,split_len):
            self.stt_data.end = split_len

            b = self.seg_buffer.to_index( self.rec_start )
            e = self.seg_buffer.to_index( split_len )
            audio = self.seg_buffer.to_numpy( b, e )
            self.stt_data.audio = audio

            b = self.hists.to_index( self.rec_start // self.frame_size )
            e = self.hists.to_index( split_len//self.frame_size )
            hists = self.hists.to_df( b, e )
            self.stt_data.hists = hists

            if self.callback is None:
                self.dict_list.append( self.stt_data )
            else:
                self.callback(self.stt_data)
    
    def end(self):
        if self.rec:
            split_len = self.seg_buffer.get_pos()
            self._flush(split_len)
            self.wave_close()

    def _wave_start(self):
        if self.wave_dir is not None:
            t:Thread = Thread( name='save', target=self._th_wave_start, daemon=True )
            t.start()

    def _wave_end(self):
        if self.wave_dir is not None:
            t:Thread = Thread( name='save', target=self._th_wave_start, daemon=True )
            t.start()

    def _th_wave_start(self):
        ws = None
        log1=0
        log2=0
        with self.wave_lock:
            if self.save_1<0:
                x:int = self.seg_buffer.offset
                data:np.ndarray = self.seg_buffer.to_numpy()
                self.save_1 = self.seg_buffer.get_pos()
                log1 = self.save_1 - len(data)
                log2 = self.save_1
                # 現在時刻のタイムスタンプを取得
                current_time = time.time() - (len(data)/self.sample_rate)
                # タイムスタンプをローカルタイムに変換して、yyyymmdd_hhmmss のフォーマットで文字列を作成
                filename = time.strftime("audio_%Y%m%d_%H%M%S.wav", time.localtime(current_time) )
            else:
                x:int = self.seg_buffer.to_index( self.save_1 )
                data:np.ndarray = self.seg_buffer.to_numpy( start=x )
                self.save_1 = self.seg_buffer.get_pos()
                log1 = self.save_1 - len(data)
                log2 = self.save_1
                for s in range(10):
                    ws = self.wave_stream
                    if ws is not None:
                        break
                    self.wave_lock.wait(1.0)
                if ws is None:
                    logger.error("can not get wave stream")
                    return
        if ws is None:
            try:
                logger.info(f"open wave file {filename}")
                print( f"[WAV]open {filename}")
                filepath = os.path.join( self.wave_dir, filename )
                ws = wave.open( filepath, 'wb')
                ws.setnchannels(1)  # モノラル
                ws.setsampwidth(2)  # サンプル幅（バイト数）
                ws.setframerate(self.sample_rate)  # サンプリングレート
                with self.wave_lock:
                    self.wave_stream = ws
            except:
                logger.exception("can not save wave")
                with self.wave_lock:
                    self.wave_dir = None
                    self.save_1 = -1
                    self.wave_stream = None
                    return
        try:
            print( f"[WAV]write [{log1}:{log2}]")
            data = data * 32767.0
            data = data.astype(np.int16).tobytes()
            ws.writeframes(data)
        except:
            logger.exception("can not save wave")
            with self.wave_lock:
                self.wave_dir = None
                self.save_1 = -1
                self.wave_stream = None

    def _wave_close(self):
        if self.wave_dir is None:
            return
        with self.wave_lock:
            if self.save_1<0:
                return
            aa = self.seg_buffer.get_pos() - self.save_1
            if aa<int(self.seg_buffer.capacity*0.5):
                return
            x:int = self.seg_buffer.to_index( self.save_1 )
            data:np.ndarray = self.seg_buffer.to_numpy( start=x )
            log2 = self.seg_buffer.get_pos()
            ws = self.wave_stream
            self.save_1 = -1
            self.wave_stream = None
            t:Thread = Thread( name='save', target=self._th_wave_close, args=(ws,data,log2), daemon=True )
            t.start()

    def _th_wave_close(self, ws, data, log2):
        try:
            log1 = log2 - len(data)
            print( f"[WAV]write [{log1}:{log2}]")
            data = data * 32767.0
            data = data.astype(np.int16).tobytes()
            ws.writeframes(data)
            print( f"[WAV]close")
            ws.close()
            logger.info( "wave file closed" )
        except:
            logger.exception("can not save wave")

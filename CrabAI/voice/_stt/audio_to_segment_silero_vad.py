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
from .hists import AudioFeatureBuffer
from .low_pos import LowPos
from ..voice_utils import voice_per_audio_rate
from .silero_vad import SileroVAD

logger = getLogger(__name__)

def rms_energy( audio, sr=16000 ):
    e = librosa.feature.rms( y=audio, hop_length=len(audio))[0][0]
    return e

def find_lowest_vad_at_slope_increase( data:np.ndarray, window_size):
    if not isinstance(data,np.ndarray):
        raise Exception("not np.ndarray")
    if len(data.shape)!=1:
        raise Exception("not np.ndarray")

    # 移動平均の計算
    conv_kernel = np.ones(window_size) / window_size
    moving_averages = np.convolve(data, conv_kernel, mode='valid')
    
    # 傾きの計算
    slopes = np.diff(moving_averages)
    
    # 傾きがプラスに変化する位置の特定
    change_points = np.where((slopes[:-1] <= 0) & (slopes[1:] > 0))[0] + 1
    if len(change_points)==0:
        return None

    # VAD評価値が最も低い点の特定
    lowest_vad = np.inf
    lowest_idx = None
    for index in change_points:
        start = max(0, index - window_size)
        end = min(index + window_size, len(data) - 1)
        idx = start + np.argmin(data[start:end+1])
        var = data[idx]
        if var < lowest_vad:
            lowest_vad = var
            lowest_idx = idx
    return lowest_idx

NON_VOICE=0
PREFIX=1
VPULSE=3
POST_VPULSE=2
PRE_VOICE=8
VOICE=9
POST_VOICE=7
TERM=4
TPULSE=6
POST_TPULSE=5

class AudioToSegmentSileroVAD:
    """音声の区切りを検出する"""
    DEFAULT_BUTTER = [ 50, 10, 10, 90 ] # fpass, fstop, gpass, gstop
    def __init__(self, *, callback, sample_rate:int=16000 ):
        # 設定
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000
        self.pick_trig:float = 0.4
        self.up_trig:float = 0.5
        self.dn_trig:float = 0.45
        self.ignore_length:int = int( 0.1 * self.sample_rate ) # 発言とみなす最低時間
        self.min_speech_length:int = int( 0.4 * self.sample_rate )
        self.max_speech_length:int = int( 4.0 * self.sample_rate )
        self.post_speech_length:int = int( 0.4 * self.sample_rate ) 
        self.max_silent_length:int = int( 0.8 * self.sample_rate )  # 発言終了とする無音時間
        self.prefech_length:int = int( 1.6 * self.sample_rate ) # 発言の途中で先行通知する時間
        self.var1 = 0.3 # 発言とみなすFFTレート
        #
        self._mute:bool = False
        self.callback = callback
        self.dict_list:list[dict] = []
        # 音声データを処理するフレームサイズに分割する
        self.frame_msec:int = 10  # 10ms,20ms,30ms
        self.frame_size:int = 512 # int( (self.sample_rate * self.frame_msec) / 1000 )  # 10ms,20ms,30ms
        self.frame_buffer_len:int = 0
        self.frame_buffer_raw:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        self.frame_buffer_cut:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        # 
        self.num_samples:int = 0
        self.last_dump:int = 0
        # AudioFeatureに必要な長さを先に計算
        self.hists:AudioFeatureBuffer = AudioFeatureBuffer( int(self.sample_rate*30/self.frame_size+0.5) )
        # AudioFeatureの長さからAudioの長さを計算
        self.seg_buffer:RingBuffer = RingBuffer( self.hists.capacity*self.frame_size, dtype=np.float32 )
        self.raw_buffer:RingBuffer = RingBuffer( self.seg_buffer.capacity, dtype=np.float32 )
        # webrtc-vad
        # SileroVAD
        self.silerovad:SileroVAD = SileroVAD( window_size_samples=self.frame_size, sampling_rate=self.sample_rate )
        #
        # 判定用 カウンタとフラグ
        self.rec:int = NON_VOICE
        self.pos:list[int] = [0] * 10
        self.rec_start:int = 0
        self.rec_end:int = 0
        self.silent_start:int = 0
        self.ignore_list:RingBuffer = RingBuffer( 10, dtype=np.int64)
        # 処理用
        self.last_down:LowPos = LowPos()
        # プリフェッチ用フラグ
        self.prefed:bool=False
        # 人の声のフィルタリング（バンドパスフィルタ）
        # self.sos = scipy.signal.butter( 4, [100, 2000], 'bandpass', fs=self.sample_rate, output='sos')
        # ハイパスフィルタ
        self._butter = AudioToSegmentSileroVAD.DEFAULT_BUTTER
        self._update_butter()

    def _update_butter(self):
        fpass, fstop, gpass, gstop = self._butter
        fn = self.sample_rate / 2   #ナイキスト周波数
        wp = fpass / fn  #ナイキスト周波数で通過域端周波数を正規化
        ws = fstop / fn  #ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
        # self.b, self.a = signal.butter(N, Wn, "high")   #フィルタ伝達関数の分子と分母を計算
        self.sos = signal.butter(N, Wn, "high", output='sos')   #フィルタ伝達関数の分子と分母を計算

    def hipass(self,x):
        #y = signal.filtfilt(self.b, self.a, x) #信号に対してフィルタをかける
        y:np.ndarray = signal.sosfiltfilt( self.sos, x ) #信号に対してフィルタをかける
        return y.astype(np.float32)

    def __getitem__(self,key):
        if 'vad.pick'==key:
            return self.pick_trig
        elif 'vad.up'==key:
            return self.up_trig
        elif 'vad.dn'==key:
            return self.dn_trig
        elif 'vad.var1'==key:
            return self.var1
        elif 'vad.butter'==key:
            return self._butter
        return None

    def to_dict(self)->dict:
        keys = ['vad.pick','vad.up','vad.dn','vad.var1','vad.butter' ]
        ret = {}
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        if 'vad.pick'==key:
            if isinstance(val,(int,float)) and 0<=val<=1:
                self.pick_trig = float(val)
        elif 'vad.up'==key:
            if isinstance(val,(int,float)) and 0<=val<=1:
                self.up_trig = float(val)
        elif 'vad.dn'==key:
            if isinstance(val,(int,float)) and 0<=val<=1:
                self.dn_trig = float(val)
        elif 'vad.var1'==key:
            if isinstance(val,(int,float)) and 0<=val<=1:
                self.var1 = float(val)
        elif 'vad.butter'==key:
            if isinstance(val,list) and len(val)==4 and all(isinstance(v, (int,float)) for v in val):
                self._butter = list(map(float, val))
                self._update_butter()

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def load(self):
        try:
            logger.info("load start")
            self.silerovad.load()
        except:
            logger.exception("")

    def start(self):
        self.frame_buffer_len=0
        self.num_samples = 0
        self.last_dump = 0

    def set_pause(self,b):
        self._mute = b

    def stop(self, *, utc:float=0.0):
        if self.rec>=PRE_VOICE:
            st_pos = self.pos[VOICE]
            end_pos = self.seg_buffer.get_pos()
            logger.info(f"[REC] stop {st_pos} {end_pos}")
            stt_data = SttData( SttData.Segment, utc, st_pos,end_pos, self.sample_rate )
            self._flush( stt_data )
        self.rec = NON_VOICE
        if self.num_samples>self.last_dump:
            sz = self.num_samples - self.last_dump
            ed = self.seg_buffer.get_pos()
            st = max(0, ed - sz)
            dmp:SttData = SttData( SttData.Dump, utc, st, ed, self.sample_rate )
            self._flush( dmp )
            self.last_dump = self.num_samples
        self.frame_buffer_len = 0
        self.num_samples = 0
        self.last_dump = 0
        self.seg_buffer.clear()
        self.raw_buffer.clear()
        self.hists.clear()

    def audio_callback(self, utc:float, raw_audio:np.ndarray, *args ) ->bool:
        """音声データをself.frame_sizeで分割して処理を呼び出す"""
        try:
            if raw_audio is None:
                # End of stream
                self.stop( utc=utc )
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
            hists_len:int = self.hists.add( frame.max(), frame.min(), self.rec, is_speech, energy, zc, 0.0 )
            hists_idx:int = hists_len - 1

            if self._mute:
                self.rec=NON_VOICE
                self.rec_start = 0

            end_pos = self.seg_buffer.get_pos()

            if self.rec==-1:
                pass

            elif self.rec==POST_VOICE:
                if is_speech>=self.up_trig:
                    # print(f"[REC] PostVoice->Voice {end_pos} {is_speech}")
                    self.last_down.push( end_pos, is_speech )
                    self.rec=VOICE
                else:
                    seg_len = end_pos - self.pos[POST_VOICE]
                    if seg_len>=self.post_speech_length:
                        # 音声終了処理
                        logger.debug(f"[REC] PostVoice->Term {end_pos} {is_speech}")
                        stt_data = SttData( SttData.Segment, utc, self.pos[VPULSE],end_pos, self.sample_rate )
                        self._flush( stt_data )
                        self.rec = TERM
                        self.pos[TERM] = end_pos

            elif self.rec==VOICE:
                if is_speech<self.dn_trig:
                    logger.debug(f"[REC] Voice->PostVoice {end_pos} {is_speech}")
                    self.rec=POST_VOICE
                    self.pos[POST_VOICE] = end_pos
                else:
                    seg_len = end_pos - self.pos[VOICE]
                    if seg_len>=self.max_speech_length:
                        # 分割処理
                        ignore = int( self.sample_rate * 0.2 )
                        st = ( self.pos[VOICE] + ignore ) // self.frame_size
                        ed = ( end_pos - ignore ) // self.frame_size
                        hist_vad = self.hists.hist_vad.to_numpy( st, ed )
                        if len(hist_vad)>0:
                            split_pos = find_lowest_vad_at_slope_increase( hist_vad, 5 )
                            if split_pos>0:
                                split_pos = (st+split_pos) * self.frame_size
                                st_sec = self.pos[VOICE]/self.sample_rate
                                ed_sec = end_pos/self.sample_rate
                                split_sec = split_pos/self.sample_rate
                                logger.debug(f"[REC] split {is_speech} {st_sec}-{ed_sec} {split_sec} {seg_len/self.sample_rate}(sec)")
                                stt_data = SttData( SttData.Segment, utc, self.pos[VPULSE],split_pos, self.sample_rate )
                                self._flush( stt_data )
                                self.pos[VPULSE] = split_pos
                                self.pos[VOICE] = split_pos
                            else:
                                logger.debug(f"[REC] failled to split ")
                        else:
                            logger.error(f"[REC] failled to split self.pos[VOICE]:{self.pos[VOICE]} end_pos:{end_pos} seg_len:{seg_len} ignore:{ignore} [{st}:{ed}]" )
                    else:
                        if is_speech<self.up_trig:
                            self.last_down.push( end_pos, is_speech )

            elif self.rec==PRE_VOICE:
                seg_len = end_pos - self.pos[PRE_VOICE]
                if seg_len>=self.min_speech_length:
                    self.rec = VOICE
                    self.pos[VOICE] = self.pos[PRE_VOICE]

            elif self.rec==VPULSE or self.rec==TPULSE:
                seg_len = end_pos - self.pos[VPULSE]
                if seg_len>=self.ignore_length or is_speech>=self.up_trig:
                    # 音声開始処理をするとこ
                    tmpbuf = self.seg_buffer.to_numpy( -seg_len )
                    var = voice_per_audio_rate(tmpbuf, sampling_rate=self.sample_rate)
                    self.hists.set_var( hists_idx, var )
                    if var>self.var1:
                        # FFTでも人の声っぽいのでセグメントとして認定
                        logger.debug( f"[REC] segment start voice/audio {var}" )
                        self.rec = PRE_VOICE
                        seg_start = self.pos[VPULSE]
                        self.pos[PRE_VOICE] = seg_start
                        # 直全のパルスをマージする
                        merge_length = int(self.sample_rate*0.4)
                        max_merge_length = int(self.sample_rate*1.2)
                        limit = seg_start - max_merge_length
                        i=len(self.ignore_list)-1
                        while i>=0 and limit<=self.ignore_list[i]:
                            if seg_start-self.ignore_list[i]<=merge_length:
                                seg_start = int(self.ignore_list[i])
                            i-=1
                        self.ignore_list.clear()
                        # 上り勾配をマージする
                        while True:
                            idx = self.hists.to_index( seg_start//self.frame_size -1 )
                            if idx<0 or self.hists.get_vad_slope( idx )<0.01:
                                break
                            seg_start -= self.frame_size
                        idx = max(0, self.hists.to_index( seg_start//self.frame_size -1 ) )
                        for idx in range( idx, hists_len):
                            hco = self.hists.hist_color[idx]
                            if hco == NON_VOICE or hco == TERM:
                                self.hists.hist_color.set(idx, PREFIX)
                        self.last_down.clear()
                        self.prefed = False
                        self.pos[VPULSE] = seg_start
                    # else:
                    #     logger.debug( f"[REC] ignore pulse voice/audio {var}" )
                elif is_speech<self.pick_trig:
                    self.rec = POST_VPULSE if self.rec==VPULSE else POST_TPULSE
                    self.pos[POST_VPULSE] = end_pos

            elif self.rec==POST_VPULSE or self.rec==POST_TPULSE:
                if is_speech>=self.pick_trig:
                    self.rec = VPULSE if self.rec==POST_VPULSE else TPULSE
                    self.ignore_list.add(end_pos)
                    self.pos[VPULSE] = end_pos
                else:
                    seg_len = end_pos - self.pos[POST_VPULSE]
                    if seg_len>=self.ignore_length:
                        #logger.debug(f"[REC] pulse->none {end_pos}")
                        self.rec = NON_VOICE if self.rec==POST_VPULSE else TERM

            elif self.rec==TERM:
                seg_len = end_pos - self.pos[TERM]
                if seg_len>=self.max_silent_length:
                    # 終了通知
                    stt_data = SttData( SttData.Term, utc, self.pos[TERM], end_pos, self.sample_rate )
                    if self.callback is not None:
                        self.callback(stt_data)
                    self.rec=NON_VOICE
                elif is_speech>=self.pick_trig:
                    # logger.debug(f"[REC] Term->T_Pulse {end_pos} {is_speech}")
                    self.rec=TPULSE
                    self.ignore_list.add(end_pos)
                    self.pos[VPULSE] = end_pos
            else:
                #NON_VOICE
                # 最後のindex
                last_idx = hists_idx - (self.hists.window//2)
                if last_idx>self.hists.window:
                    slope0 = self.hists.get_vad_slope(last_idx-1)
                    slope1 = self.hists.get_vad_slope(last_idx)
                    vx = 0.05
                    if vx>slope0 and vx<=slope1:
                        xx_pos = end_pos - (self.frame_size*(1+self.hists.window//2))
                        self.ignore_list.add( xx_pos )
                        logger.debug(f"[REC] pulse {xx_pos} slope:{slope0:.3f} {slope1:.3f}")
                if is_speech>=self.pick_trig:
                    # logger.debug(f"[REC] NoVice->V_Pulse {end_pos} {is_speech}")
                    self.rec=VPULSE
                    self.ignore_list.add(end_pos)
                    self.pos[VPULSE] = end_pos

            self.hists.set_color( hists_idx, self.rec )
            if (num_samples-self.last_dump)>=self.seg_buffer.capacity:
                self.last_dump = num_samples
                ed = self.seg_buffer.get_pos()
                st = max(0, ed - self.seg_buffer.capacity)
                dmp:SttData = SttData( SttData.Dump, utc, st, ed, self.sample_rate )
                self._flush( dmp )
        except:
            logger.exception(f"")

    def _flush(self,stt_data:SttData):
            start_pos = stt_data.start
            end_pos = stt_data.end

            b = self.seg_buffer.to_index( start_pos )
            e = self.seg_buffer.to_index( end_pos )
            raw = self.raw_buffer.to_numpy( b, e )
            stt_data.raw = raw
            audio = self.hipass(raw)
            stt_data.audio = audio

            b = self.hists.to_index( start_pos // self.frame_size )
            e = self.hists.to_index( end_pos//self.frame_size )
            hists = self.hists.to_df( b, e )
            stt_data.hists = hists

            if self.callback is None:
                self.dict_list.append( stt_data )
            else:
                self.callback(stt_data)
    
    def end(self, *, utc:float=0.0):
        self.stop( utc=utc )

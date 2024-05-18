import sys,os
import platform
from logging import getLogger
import time
import numpy as np
from multiprocessing.queues import Queue
import wave
import sounddevice as sd
import librosa
from scipy import signal

from CrabAI.vmp import Ev, VFunction, VProcess
from .mic_to_audio import get_mic_devices
from .stt_data import SttData
from .ring_buffer import RingBuffer
from .hists import AudioFeatureBuffer
from .low_pos import LowPos
from ..voice_utils import voice_per_audio_rate
from .silero_vad import SileroVAD

logger = getLogger(__name__)

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

class AudioToSegment(VFunction):

    def __init__(self, data_in:Queue, data_out:Queue, ctl_out:Queue, *, sample_rate:int ):

        super().__init__(data_in,data_out,ctl_out)
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

        self.frame_size:int = 512
        self.buffer_sec:float = 31.0
        # AudioFeatureに必要な長さを先に計算
        self.hists:AudioFeatureBuffer = AudioFeatureBuffer( int(self.sample_rate*self.buffer_sec/self.frame_size+0.5) )
        # AudioFeatureの長さからAudioの長さを計算
        self.seg_buffer:RingBuffer = RingBuffer( self.hists.capacity*self.frame_size, dtype=np.float32 )
        self.raw_buffer:RingBuffer = RingBuffer( self.seg_buffer.capacity, dtype=np.float32 )
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

        # dump
        self.dump_last_fr:int = 0
        self.dump_interval_sec:float = 30.0

    def load(self):
        pass

    def proc(self,ev:Ev):
        if isinstance(ev,SttData):
            self.proc_stt_data(ev)

    def stop(self):
        self.proc_output_dump( 0,True)

    def proc_stt_data(self,stt_data:SttData):

        audio_length:int = stt_data.audio.shape[0]
        hist_length:int = stt_data.hists.shape[0]
        seglen:int = int( audio_length/hist_length )
        if int(audio_length/seglen) != hist_length:
            raise ValueError(f"ERROR?")
        if seglen != self.frame_size:
            raise ValueError(f"ERROR")
        if self.sample_rate != stt_data.sample_rate:
            raise ValueError(f"ERROR")

        utc:float = stt_data.utc

        pos = stt_data.start
        for s in range(0,audio_length,seglen):
            raw = stt_data.raw[s:s+seglen]
            audio = stt_data.audio[s:s+seglen]
            xs = int(s/seglen)
            hi = stt_data.hists['hi'][xs]
            lo = stt_data.hists['lo'][xs]

            vad = stt_data.hists['vad'][xs]
            vad_ave = stt_data.hists['vad_ave'][xs]
            energy = stt_data.hists['energy'][xs]
            zc = stt_data.hists['zc'][xs]
            var = stt_data.hists['var'][xs]
            mute = stt_data.hists['mute'][xs]

            self.proc_audio_data( utc, pos+s, raw, audio, hi, lo, vad, vad_ave, energy, zc, var, mute )

    def proc_audio_data(self, utc, pos, frame_raw, frame, hi, lo, vad, vad_ave, energy, zc, var, mute ):
            self.seg_buffer.append(frame)
            self.raw_buffer.append(frame_raw)
            hists_len:int = self.hists.put( hi,lo, self.rec, vad, vad_ave, energy, zc, var, mute )
            hists_idx:int = hists_len - 1
            if hists_len<5:
                return

            is_speech:float = vad_ave

            if mute:
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
                    self.proc_output_event(stt_data)
                    self.rec=NON_VOICE
                elif is_speech>=self.pick_trig:
                    # logger.debug(f"[REC] Term->T_Pulse {end_pos} {is_speech}")
                    self.rec=TPULSE
                    self.ignore_list.add(end_pos)
                    self.pos[VPULSE] = end_pos
            else:
                #NON_VOICE
                # 最後のindex
                slope0 = self.hists.get_vad_slope(hists_len-2)
                slope1 = self.hists.get_vad_slope(hists_len-1)
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
            self.proc_output_dump(utc,False)

    def _flush(self,stt_data:SttData):
            start_pos = stt_data.start
            end_pos = stt_data.end

            b = self.seg_buffer.to_index( start_pos )
            e = self.seg_buffer.to_index( end_pos )
            stt_data.raw = self.raw_buffer.to_numpy( b, e )
            stt_data.audio = self.seg_buffer.to_numpy( b, e )

            b = self.hists.to_index( start_pos // self.frame_size )
            e = self.hists.to_index( end_pos//self.frame_size )
            hists = self.hists.to_df( b, e )
            stt_data.hists = hists

            self.proc_output_event( stt_data )

    def to_stt_data(self, typ:int, utc:float, start_fr:int, end_fr:int ) ->None:
        st = start_fr * self.frame_size
        ed = end_fr * self.frame_size
        stt_data = SttData( typ, utc, st, ed, self.sample_rate )
        b = self.seg_buffer.to_index( st )
        e = self.seg_buffer.to_index( ed )
        stt_data.raw = self.raw_buffer.to_numpy( b, e )
        stt_data.audio = self.seg_buffer.to_numpy( b, e )

        b = self.hists.to_index( start_fr )
        e = self.hists.to_index( end_fr )
        stt_data.hists = self.hists.to_df( b, e )
        return stt_data

    def proc_output_dump(self,utc:float,flush:bool):
        try:
            if self.dump_interval_sec<=0:
                return
            last_fr = self.hists.get_pos()
            if self.dump_last_fr>=last_fr:
                return
            interval_fr:int = int( (self.sample_rate*self.dump_interval_sec) / self.frame_size )
            if not flush and (last_fr-self.dump_last_fr)<interval_fr:
                return
            stt_data:SttData = self.to_stt_data( SttData.Dump, utc, self.dump_last_fr, last_fr )
            self.dump_last_fr = last_fr
            self.output_ctl( stt_data )
        finally:
            pass
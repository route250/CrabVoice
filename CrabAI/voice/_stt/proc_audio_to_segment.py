import sys,os
import platform
from logging import getLogger
import time
import numpy as np
from multiprocessing.queues import Queue as PQ
import wave
import sounddevice as sd
import librosa
from scipy import signal

from CrabAI.vmp import Ev, ShareParam, VFunction, VProcess
from .stt_data import SttData
from .ring_buffer import RingBuffer
from .hists import AudioFeatureBuffer
from ..voice_utils import voice_per_audio_rate

logger = getLogger(__name__)

def find_lowest_vad_at_slope_increase( vad_averages:np.ndarray, vad:np.ndarray, energy:np.ndarray, window_size:int):
    """セグメントを分割する位置を決める"""
    if not isinstance(vad,np.ndarray):
        raise Exception("not np.ndarray")
    if len(vad.shape)!=1:
        raise Exception("not np.ndarray")

    # 傾きの計算
    slopes = np.diff(vad_averages)
    
    # 傾きがプラスに変化する位置の特定
    threashold:float = 0.05
    change_points = np.where((slopes[:-1] <= threashold) & (slopes[1:] > threashold))[0] + 1
    if len(change_points)==0:
        return None

    # VAD評価値が最も低い点の特定
    lowest_vad = np.inf
    vad_idx = None
    for index in change_points:
        start = max(0, index - window_size)
        end = min(index + window_size+1, len(vad))
        idx = start + np.argmin(vad[start:end])
        value = vad[idx]
        if value < lowest_vad:
            lowest_vad = value
            vad_idx = idx
    if lowest_vad<0.5:
        return vad_idx

    # energy評価値が最も低い点の特定
    lowest_energy = np.inf
    energy_idx = None
    for index in change_points:
        start = max(0, index - window_size)
        end = min(index + window_size+1, len(vad))
        idx = start + np.argmin(energy[start:end])
        value = energy[idx]
        if value < lowest_energy:
            lowest_energy = value
            energy_idx = idx
    if energy_idx<vad_idx:
        return energy_idx
    else:
        return vad_idx

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

    @staticmethod
    def load_default( conf:ShareParam ):
        if isinstance(conf,ShareParam):
            conf.set_vad_pick(0.4 )
            conf.set_vad_up(0.5 )
            conf.set_vad_dn(0.45 )
            conf.set_vad_ignore_sec(0.1 )
            conf.set_vad_min_sec(0.4 )
            conf.set_vad_max_sec(4.0 )
            conf.set_vad_post_sec(0.2 )
            conf.set_vad_silent_sec(0.8 )
            conf.set_vad_var( 0.3 )
            conf.set_aux( 0, 0, 0, 0, 0 )

    def __init__(self, proc_no:int, num_proc:int, conf:ShareParam, data_in:PQ, data_out:PQ, ctl_out:PQ, *, sample_rate:int ):

        super().__init__(proc_no,num_proc,conf,data_in,data_out)
        self.ctl_out:PQ = ctl_out
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000

        self.pick_trig:float = None
        self.up_trig:float = None
        self.dn_trig:float = None
        self.ignore_length:int = None
        self.min_speech_length:int = None
        self.max_speech_length:int = None
        self.post_speech_length:int = None
        self.max_silent_length:int = None
        self.var1:float = None
        self.prefech_length:int = None
        self.reload_share_param()

        # dump
        self.last_utc:float = 0
        self.dump_last_fr:int = 0
        self.dump_interval_sec:float = 30.0

        self.frame_size:int = 512
        buffer_sec:float = self.dump_interval_sec + 1.0
        # AudioFeatureに必要な長さを先に計算
        self.hists:AudioFeatureBuffer = AudioFeatureBuffer( int(self.sample_rate*buffer_sec/self.frame_size+0.5) )
        # AudioFeatureの長さからAudioの長さを計算
        self.seg_buffer:RingBuffer = RingBuffer( self.hists.capacity*self.frame_size, dtype=np.float32 )
        self.raw_buffer:RingBuffer = RingBuffer( self.seg_buffer.capacity, dtype=np.float32 )
        #
        # 判定用 カウンタとフラグ
        self.rec:int = NON_VOICE
        self.rec_max = self.max_speech_length
        self.rec_start:int = 0
        self.pos_SEGSTART:int = 0
        self.pos_VOICE:int = 0
        self.silent_start:int = 0
        self.ignore_list:RingBuffer = RingBuffer( 10, dtype=np.int64)
        # 処理用
        # プリフェッチ用フラグ
        self.prefed:bool=False

        # fade in/out
        self.fade_frames = int(self.sample_rate*0.05/self.frame_size+1)
        self.fade_len:int = self.fade_frames * self.frame_size
        w = signal.windows.hann(self.fade_len*2)
        self.fade_in_window = w[:self.fade_len]
        self.fade_out_window = w[self.fade_len:]

        # ログ用
        self.mute:bool=False

    def load(self):
        pass

    def reload_share_param(self):
        self.pick_trig = self.conf.get_vad_pick()
        self.up_trig = self.conf.get_vad_up()
        self.dn_trig = self.conf.get_vad_dn()
        self.ignore_length = int( self.conf.get_vad_ignore_sec() * self.sample_rate ) # 発言とみなす最低時間
        self.min_speech_length = int( self.conf.get_vad_min_sec() * self.sample_rate )
        self.max_speech_length = int( self.conf.get_vad_max_sec() * self.sample_rate )
        self.post_speech_length = int( self.conf.get_vad_post_sec() * self.sample_rate ) 
        self.max_silent_length = int( self.conf.get_vad_silent_sec() * self.sample_rate )  # 発言終了とする無音時間
        self.prefech_length = int( 1.6 * self.sample_rate ) # 先行通知する時間
        self.var1 = self.conf.get_vad_var()

    def proc(self,ev:Ev):
        if isinstance(ev,SttData):
            self.proc_stt_data(ev)
        elif ev.typ == Ev.EndOfData or ev.typ == Ev.Stop:
            self.proc_end_of_data()

    def stop(self):
        pass

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
            self.last_utc = utc
            self.seg_buffer.append(frame)
            self.raw_buffer.append(frame_raw)

            if not self.mute:
                if mute:
                    self.mute=True
                    print("### MUTE TRUE ###")
                    self.rec=NON_VOICE
                    self.rec_start = 0
                    self.pos_SEGSTART = 0
                    self.pos_VOICE = 0
                    self.ignore_list.clear()
            else:
                if not mute and vad_ave<self.dn_trig:
                    self.mute = False
                    print("### MUTE FALSE ###")

            current_pos:int = self.seg_buffer.get_pos()
            hists_len:int = self.hists.put( hi,lo, self.rec, vad, vad_ave, energy, zc, var, self.mute )
            hists_idx:int = hists_len - 1
            if hists_len<5:
                return

            is_speech:float = vad_ave
            end_pos = self.seg_buffer.get_pos()

            if self.rec==-1 or self.mute:
                pass

            elif self.rec==POST_VOICE:
                if is_speech>=self.up_trig:
                    # print(f"[REC] PostVoice->Voice {end_pos} {is_speech}")
                    self.rec=VOICE
                    self.rec_start = current_pos
                else:
                    seg_len = end_pos - self.rec_start
                    if seg_len>=self.post_speech_length:
                        # 音声終了処理
                        logger.debug(f"[REC] PostVoice->Term {end_pos} {is_speech}")
                        self.output_audio_segment( SttData.Segment, utc, self.pos_SEGSTART,end_pos )
                        self.rec = TERM
                        self.rec_start = end_pos

            elif self.rec==VOICE:
                if is_speech<self.dn_trig:
                    logger.debug(f"[REC] Voice->PostVoice {end_pos} {is_speech}")
                    self.rec=POST_VOICE
                    self.rec_start = end_pos
                else:
                    seg_len = end_pos - self.pos_VOICE
                    if seg_len>=self.rec_max:
                        # 分割処理
                        ignore = int( self.sample_rate * 0.2 )
                        st_fr = ( self.pos_VOICE + ignore ) // self.frame_size
                        sti = self.hists.hist_vad.to_index( st_fr )
                        ed_fr = ( end_pos - ignore ) // self.frame_size
                        edi = self.hists.hist_vad.to_index( ed_fr )
                        hist_vad_ave = self.hists.hist_vad.to_numpy( sti, edi )
                        hist_vad = self.hists.hist_vad.to_numpy( sti, edi )
                        hist_energy = self.hists.hist_energy.to_numpy( sti, edi )
                        self.rec_max += self.min_speech_length
                        if len(hist_vad_ave)>0:
                            split_idx = find_lowest_vad_at_slope_increase( hist_vad_ave, hist_vad, hist_energy,5 )
                            if split_idx is not None and split_idx>0:
                                split_fr = st_fr + split_idx
                                split_idx = self.hists.hist_var.to_index(split_fr)
                                self.hists.set_color(split_idx-1, POST_VOICE )
                                split_pos = split_fr * self.frame_size
                                st_sec = self.pos_VOICE/self.sample_rate
                                ed_sec = end_pos/self.sample_rate
                                split_sec = split_pos/self.sample_rate
                                logger.debug(f"[REC] split {is_speech} {st_sec}-{ed_sec} {split_sec} {seg_len/self.sample_rate}(sec)")
                                self.output_audio_segment( SttData.Segment, utc, self.pos_SEGSTART,split_pos )
                                self.rec_start = split_pos
                                self.pos_SEGSTART = split_pos
                                self.pos_VOICE = split_pos
                                self.rec_max = self.max_speech_length
                            else:
                                logger.debug(f"[REC] failled to split ")
                        else:
                            logger.error(f"[REC] failled to split self.pos_VOICE:{self.pos_VOICE} end_pos:{end_pos} seg_len:{seg_len} ignore:{ignore} [{st_fr}:{ed_fr}]" )

            elif self.rec==PRE_VOICE:
                seg_len = end_pos - self.rec_start
                if seg_len>=self.min_speech_length:
                    self.rec = VOICE
                    self.pos_VOICE = self.rec_start
                    self.rec_start = self.rec_start
                    self.rec_max = self.max_speech_length

            elif self.rec==VPULSE or self.rec==TPULSE:
                seg_len = end_pos - self.rec_start
                if seg_len>=self.ignore_length or is_speech>=self.up_trig:
                    # 音声開始処理をするとこ
                    tmpbuf = self.seg_buffer.to_numpy( -seg_len )
                    var = voice_per_audio_rate(tmpbuf, sampling_rate=self.sample_rate)
                    self.hists.set_var( hists_idx, var )
                    if var>self.var1:
                        # FFTでも人の声っぽいのでセグメントとして認定
                        logger.debug( f"[REC] segment start voice/audio {var}" )
                        self.rec = PRE_VOICE
                        seg_start = self.rec_start
                        self.rec_start = seg_start
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
                        self.prefed = False
                        self.pos_SEGSTART = seg_start
                    # else:
                    #     logger.debug( f"[REC] ignore pulse voice/audio {var}" )
                elif is_speech<self.pick_trig:
                    self.rec = POST_VPULSE if self.rec==VPULSE else POST_TPULSE
                    self.rec_start = current_pos

            elif self.rec==POST_VPULSE or self.rec==POST_TPULSE:
                if is_speech>=self.pick_trig:
                    self.rec = VPULSE if self.rec==POST_VPULSE else TPULSE
                    self.rec_start = current_pos
                    self.ignore_list.add(current_pos)
                else:
                    seg_len = end_pos - self.rec_start
                    if seg_len>=self.ignore_length:
                        #logger.debug(f"[REC] pulse->none {end_pos}")
                        self.rec = NON_VOICE if self.rec==POST_VPULSE else TERM
                        self.rec_start = current_pos

            elif self.rec==TERM:
                seg_len = end_pos - self.rec_start
                if seg_len>=self.max_silent_length:
                    # 終了通知
                    stt_data = SttData( SttData.Term, utc, self.rec_start, end_pos, self.sample_rate )
                    self.proc_output_event(stt_data)
                    self.rec=NON_VOICE
                    self.rec_start = 0
                elif is_speech>=self.pick_trig:
                    # logger.debug(f"[REC] Term->T_Pulse {end_pos} {is_speech}")
                    self.rec=TPULSE
                    self.rec_start = current_pos
                    self.ignore_list.add(current_pos)
            else:
                #NON_VOICE
                if is_speech>=self.pick_trig:
                    # logger.debug(f"[REC] NoVice->V_Pulse {end_pos} {is_speech}")
                    self.rec=VPULSE
                    self.rec_start = current_pos
                    self.ignore_list.add(current_pos)
                else:
                    slope0 = self.hists.get_vad_slope(hists_idx-1)
                    slope1 = self.hists.get_vad_slope(hists_idx)
                    vx = 0.05
                    if vx>slope0 and vx<=slope1:
                        self.ignore_list.add( current_pos )
                        logger.debug(f"[REC] pulse {current_pos} slope:{slope0:.3f} {slope1:.3f}")

            self.hists.set_color( hists_idx, self.rec )
            self.proc_output_dump(utc,False)
            self.conf.set_aux( self.rec, vad, energy, zc, self.mute )

    def proc_end_of_data(self):
        utc = self.last_utc
        end_pos = self.seg_buffer.get_pos()
        if (self.rec == POST_VOICE or self.rec == VOICE) and (end_pos-self.pos_SEGSTART)>=self.min_speech_length:
            # 音声終了処理
            logger.debug(f"[REC] PostVoice->Term {end_pos}")
            self.output_audio_segment( SttData.Segment, utc, self.pos_SEGSTART,end_pos )
            # 終了通知
            stt_data = SttData( SttData.Term, utc, end_pos, end_pos, self.sample_rate )
            self.proc_output_event(stt_data)
            self.rec=NON_VOICE
            self.rec_start = end_pos
        self.proc_output_dump( utc,True) 

    def output_audio_segment(self, typ, utc, ss, ee ):
            f1 = len(self.fade_in_window)
            f2 = len(self.fade_out_window)
            start_pos = max( ss - f1, self.seg_buffer.to_pos(0) ) # 前余白
            end_pos = min( ee + f2, self.seg_buffer.get_pos() ) # 後ろ余白
            stt_data = SttData( typ, utc, start_pos, end_pos, self.sample_rate )

            start_idx = self.seg_buffer.to_index( start_pos )
            end_idx = self.seg_buffer.to_index( end_pos )
            audio = self.raw_buffer.to_numpy( start_idx, end_idx )
            audio[:f1] = audio[:f1] * self.fade_in_window
            audio[-f2:] = audio[-f2:] * self.fade_out_window
            stt_data.raw = audio
            audio = self.seg_buffer.to_numpy( start_idx, end_idx )
            audio[:f1] = audio[:f1] * self.fade_in_window
            audio[-f2:] = audio[-f2:] * self.fade_out_window
            stt_data.audio = audio

            start_idx = self.hists.to_index( start_pos // self.frame_size )
            end_idx = self.hists.to_index( end_pos//self.frame_size )
            hists = self.hists.to_df( start_idx, end_idx )
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

    def output_ctl(self, ev:Ev):
        if isinstance(self.ctl_out,PQ):
            ev.proc_no=self.proc_no
            ev.num_proc=self.num_proc
            self.ctl_out.put(ev)

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
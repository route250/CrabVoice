from logging import getLogger
import time
import numpy as np
from multiprocessing.queues import Queue
from queue import Empty
import librosa
from scipy import signal

from CrabAI.vmp import Ev, ShareParam, VFunction, VProcess
from .stt_data import SttData
from .ring_buffer import RingBuffer
from .hists import AudioFeatureBuffer
from .silero_vad import SileroVAD

logger = getLogger(__name__)

def rms_energy( audio, sr=16000 ):
    e = librosa.feature.rms( y=audio, hop_length=len(audio))[0][0]
    return e

def pad_to_length( arr, length):
    if len(arr) < length:
        pad_width = length - len(arr)
        padded_arr = np.pad(arr, (0, pad_width), mode='constant', constant_values=0)
        return padded_arr
    else:
        return arr

def shrink( data:np.ndarray, l:int ):
    step = int( len(data)// l )
    if step*l == len(data):
        return data[0:len(data):step]
    a = data[:l].copy()
    h = int(l//2)
    a[-h:] = data[-h:]
    return a

class SourceToAudio(VFunction):
    DEFAULT_BUTTER = tuple( [50, 10, 10, 90] ) # fpass, fstop, gpass, gstop
    @staticmethod
    def load_default( conf:ShareParam ):
        if isinstance(conf,ShareParam):
            conf.set_audio_butter(SourceToAudio.DEFAULT_BUTTER)

    def __init__(self, proc_no:int, num_proc:int, conf:ShareParam, data_in:Queue, data_out:Queue, sample_rate:int=None ):
        super().__init__(proc_no,num_proc,conf,data_in,data_out)
        self.state:int = 0
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000
        self.segsize:int = 0
        # for filter
        self.filt_mute:bool = True
        self.filt_utc:float = 0
        self.filt_buf:np.ndarray = np.zeros( 0, dtype=np.float32)
        self.fade_in_window = None
        self.fade_out_window = None
        self._butter = SourceToAudio.DEFAULT_BUTTER
        # for split
        # 音声データを処理するフレームサイズに分割する
        self.frame_msec:int = 10  # 10ms,20ms,30ms
        self.frame_size:int = 512 # int( (self.sample_rate * self.frame_msec) / 1000 )  # 10ms,20ms,30ms
        self.frame_buffer_len:int = 0
        self.frame_buffer_raw:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        self.frame_buffer_cut:np.ndarray = np.zeros( self.frame_size, dtype=np.float32)
        self.num_samples:int = 0
        #
        self.buffer_sec:float = 31.0
        # AudioFeatureに必要な長さを先に計算
        self.hists:AudioFeatureBuffer = AudioFeatureBuffer( int(self.sample_rate*self.buffer_sec/self.frame_size+0.5) )
        # AudioFeatureの長さからAudioの長さを計算
        self.seg_buffer:RingBuffer = RingBuffer( self.hists.capacity*self.frame_size, dtype=np.float32 )
        self.raw_buffer:RingBuffer = RingBuffer( self.seg_buffer.capacity, dtype=np.float32 )
        # SileroVAD
        self.silerovad:SileroVAD = SileroVAD( window_size_samples=self.frame_size, sampling_rate=self.sample_rate )

        # output
        self.out_frames:int = int( self.sample_rate * 0.1 / self.frame_size )
        self.out_last_fr:int = 0

        # sos
        self.orig_sr:int = -1
        self.sos = None

        #
        self.reload_share_param()

    def _update_butter(self, orig_sr):
        if isinstance(orig_sr,int) and orig_sr>=16000 and self.sos is None or self.orig_sr != orig_sr:
            fpass, fstop, gpass, gstop = self._butter
            fn = orig_sr / 2   #ナイキスト周波数
            wp = fpass / fn  #ナイキスト周波数で通過域端周波数を正規化
            ws = fstop / fn  #ナイキスト周波数で阻止域端周波数を正規化
            N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
            self.sos = signal.butter(N, Wn, "high", output='sos')   #フィルタ伝達関数の分子と分母を計算

    def hipass(self,x):
        if self.sos is None:
            return x + 0.1
        else:
            #y = signal.filtfilt(self.b, self.a, x) #信号に対してフィルタをかける
            y:np.ndarray = signal.sosfiltfilt( self.sos, x ) #信号に対してフィルタをかける
            return y.astype(np.float32)

    def load(self):
        try:
            librosa.resample( np.zeros( (1,1), dtype=np.float32), orig_sr=self.sample_rate*2, target_sr=self.sample_rate ) # preload of librosa
        except:
            logger.exception("load librosa.resample")

        try:
            self.silerovad.load()
            self.silerovad.is_speech( np.zeros( self.frame_size, dtype=np.float32))
        except:
            logger.exception("load self.silerovad")

    def reload_share_param(self):
        butter = self.conf.get_audio_butter()
        if isinstance(butter,list) and len(butter)==len(self._butter) and butter != self._butter:
            self._butter = butter
            self._update_butter( self.orig_sr )

    def proc( self, ev:Ev ):
        try:
            if isinstance(ev,SttData):
                stt_data: SttData = ev
                utc = stt_data.utc
                raw = stt_data.raw
                # print(f"audio proc {raw.shape} {raw.dtype}")
                mute = False
                if stt_data.hists is not None:
                    mute = True
                orig_sr = stt_data.sample_rate
                if orig_sr is None:
                    print(f"ERROR: can not open mic")
                if self.state==0:
                    self.state=1
                    self.frame_buffer_len = 0
                    self.filt_utc = -1 # フィルタ処理の先頭を無視する指定
                    self.out_last_fr = 0
                    self.orig_sr = orig_sr
                    self.segsize = self._input_seg_size(orig_sr)
                    self._update_butter( orig_sr )
                    self.proc_audio_filter( -1, np.zeros(self.segsize, dtype=np.float32), True, orig_sr )
                self.proc_audio_filter( utc, raw, mute, orig_sr )
            else:
                if self.state!=0 and ev.typ == Ev.EndOfData:
                    self.proc_audio_filter( -2, np.zeros(self.segsize, dtype=np.float32), True, self.orig_sr )
                    self.state=0
        except:
            logger.exception("proc")

    def _input_seg_size(self, orig_sr:int ) -> int:
        segsize = int( self.sample_rate * ( orig_sr/self.sample_rate ) * 0.1 )
        return segsize

    def proc_audio_filter(self, utc:float, raw_audio:np.ndarray, mute:bool, orig_sr:int):
        """音声データにフィルタを適用して次の処理へ"""
        try:
            ll:int = len(raw_audio)
            half:int = ll//2
            if ll*2 != len(self.filt_buf):
                self.filt_utc = -1
                self.filt_mute = True
                self.filt_buf:np.ndarray = np.zeros( ll+half*2, dtype=np.float32)
                self.fade_in_window = signal.windows.hann(half*2)[:half]
                self.fade_out_window = signal.windows.hann(half*2)[half:]
                print(f"filt rawlen:{ll} half:{half} buflen:{len(self.filt_buf)}")
            p_main:int = half
            p_next:int = p_main+ll
            p_center:int = p_next - half
            p_end:int = p_next + half

            # ウインドウ関数fade_outを適用してnext領域にコピー
            self.filt_buf[p_next:p_end] = raw_audio[:half] * self.fade_out_window

            # 低周波数カットフィルタの適用
            if self.sos is not None:
                segment0_filtered = self.hipass(self.filt_buf) # ローカットフィルタ
            else:
                segment0_filtered = self.filt_buf + 0.1 # debug用

            # 真ん中だけ抜き出して次の処理をコール
            raw = self.filt_buf[p_main:p_next]
            filtered = segment0_filtered[p_main:p_next]
            if orig_sr != self.sample_rate:
                raw = librosa.resample( raw, axis=0, orig_sr=orig_sr, target_sr=self.sample_rate )
                filtered = librosa.resample( filtered, axis=0, orig_sr=orig_sr, target_sr=self.sample_rate )
                if self.sos is None:
                    print(f"[filt] seglen:{len(raw)}")
                    raw = shrink( self.filt_buf[p_main:p_next], len(raw) )
                    filtered = shrink( segment0_filtered[p_main:p_next], len(filtered) )

            if self.filt_utc>=0.0:
                self.proc_audio_split( self.filt_utc, self.filt_mute, raw, filtered )
            # 真ん中のデータの後半にウインドウ関数fade_inを適用してbefore領域にコピー
            self.filt_buf[0:p_main] = self.filt_buf[p_center:p_next] * self.fade_in_window
            #
            self.filt_buf[p_main:p_next] = raw_audio
            self.filt_utc = utc
            self.filt_mute = mute
        except:
            logger.exception(f"")

    def proc_audio_split(self, utc:float, mute:bool, raw_audio:np.ndarray, audio_data:np.ndarray ) ->None:
        """音声データをself.frame_sizeで分割して処理を呼び出す"""
        try:
            buffer_raw:np.ndarray = self.frame_buffer_raw
            buffer_cut:np.ndarray = self.frame_buffer_cut
            buffer_len:int = self.frame_buffer_len

            audio_len = len(raw_audio)
            pos = 0
            while pos<audio_len:
                # 分割
                nn = min( audio_len-pos, self.frame_size - buffer_len )
                np.copyto( buffer_raw[buffer_len:buffer_len+nn], raw_audio[pos:pos+nn])
                np.copyto( buffer_cut[buffer_len:buffer_len+nn], audio_data[pos:pos+nn])
                buffer_len += nn
                pos+=nn
                # framesizeになったら呼び出す
                if buffer_len>=self.frame_size:
                    self._Process_frame( utc, mute, buffer_raw, buffer_cut )
                    self.num_samples += buffer_len
                    buffer_len = 0
            self.frame_buffer_len = buffer_len
        except:
            logger.exception(f"")

    def _Process_frame(self, utc:float, mute:bool, frame_raw:np.ndarray, frame:np.ndarray ):
        try:
            if len(frame_raw) != self.frame_size or len(frame) != self.frame_size:
                raise ValueError( f"invalid frame length. self.frame_size={self.frame_size} raw={len(frame_raw)} frame={len(frame)}" )

            # Audioデータ
            self.seg_buffer.append(frame)
            self.raw_buffer.append(frame_raw)
            #---
            # 評価値を計算
            #---
            # vadカウンタ                
            vad:float = self.silerovad.is_speech( frame )
            # ゼロ交錯数
            zz = librosa.zero_crossings(frame)
            zc = sum(zz)
            # エナジー
            energy = rms_energy(frame, sr=self.sample_rate )
            self.hists.add( frame.max(), frame.min(), 0, vad, energy, zc, 0.0, mute )

            if self.seg_buffer.get_pos() != self.raw_buffer.get_pos():
                raise Exception( f"Internal error! missmatch pos in seg and raw")
            if self.seg_buffer.get_pos() != ( self.hists.get_pos()*self.frame_size):
                raise Exception( f"Internal error! missmatch pos in seg and hists")

            window_offset = int( self.hists.window//2 )
            last_fr = self.hists.get_pos() - window_offset

            start_fr = self.out_last_fr
            end_fr = start_fr + self.out_frames
            if end_fr<=last_fr:
                stt_data:SttData = self.to_stt_data( SttData.Audio, utc, start_fr, end_fr )
                self.proc_output_event( stt_data )
                self.out_last_fr = end_fr
        except:
            logger.exception(f"")

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



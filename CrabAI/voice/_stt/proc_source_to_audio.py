import sys,os
import platform
from logging import getLogger
import time
import numpy as np
from multiprocessing.queues import Queue
from queue import Empty
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
    DEFAULT_BUTTER = [ 50, 10, 10, 90 ] # fpass, fstop, gpass, gstop
    def __init__(self, proc_no:int, num_proc:int, data_in:Queue, data_out:Queue, ctl_out:Queue, *, mic=None, source=None, sample_rate:int=None ):
        super().__init__(proc_no,num_proc,data_in,data_out,ctl_out)
        self.enable_in = False
        self.state:int = 0
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000
        self.mic_index = mic
        self.mic_name = None
        self.mic_sampling_rate = None
        self.audioinput:sd.InputStream = None
        self.mic_mute:bool = False
        #
        self.source_file_path = source
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

    def _update_butter(self, orig_sr):
        fpass, fstop, gpass, gstop = self._butter
        fn = orig_sr / 2   #ナイキスト周波数
        wp = fpass / fn  #ナイキスト周波数で通過域端周波数を正規化
        ws = fstop / fn  #ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
        # self.b, self.a = signal.butter(N, Wn, "high")   #フィルタ伝達関数の分子と分母を計算
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
            logger.exception("load")

        self.silerovad.load()
        self.silerovad.is_speech( np.zeros( self.frame_size, dtype=np.float32))

        if self.source_file_path is not None:
            self.load_file()
        elif self.mic_index is not None:
            self.load_mic()

    def load_file(self):
        pass

    def load_mic(self):
        if self.mic_index is not None:
            inp_dev_list = get_mic_devices(samplerate=self.sample_rate, dtype=np.float32,check=False)
            if inp_dev_list and len(inp_dev_list)>0:
                self.mic_index='default'
                if self.mic_index=='' or self.mic_index=='default':
                    self.mic_index = inp_dev_list[0]['index']
                    self.mic_name = inp_dev_list[0]['name']
                    self.mic_sampling_rate = int( inp_dev_list[0].get('default_samplerate',self.sample_rate) )
                    logger.info(f"selected mic : {self.mic_index} {self.mic_name} {self.mic_sampling_rate}")
                else:
                    for m in inp_dev_list:
                        if m['index'] == self.mic_index:
                            self.mic_sampling_rate = int( m.get('default_samplerate',self.sample_rate) )
                            self.mic_name = m.get('name','mic')
                print(f"selected mic : {self.mic_index} {self.mic_name} {self.mic_sampling_rate}")
            else:
                self.mic_index = None
        elif self.file_path is not None:
            pass

    def proc(self, ev ):
        try:
            self.frame_buffer_len = 0
            self.filt_utc = -1 # フィルタ処理の先頭を無視する指定
            self.out_last_fr = 0
            if self.source_file_path is not None:
                if self.source_file_path.endswith('.wav'):
                    self.proc_wav_file()
                elif self.source_file_path.endswith('.npz'):
                    self.start_stt_data_file()
            elif self.mic_index is not None:
                self.proc_mic()
            else:
                self.proc_debug_data()
        finally:
            pass

        # try:
        #     if self.dump_interval_sec>0:
        #         audio_sec:float = (end_pos - self.dump_last_pos)/self.sample_rate
        #         if audio_sec>=self.dump_interval_sec:
        #             stt_data:SttData = self.to_stt_data( SttData.Dump, utc, self.dump_last_pos, end_pos )
        #             self.dump_last_pos = end_pos
        #             self.output_ctl( stt_data )

    def _input_seg_size(self, orig_sr:int ) -> int:
        segsize = int( self.sample_rate * ( orig_sr/self.sample_rate ) * 0.1 )
        return segsize

    def proc_ctl(self):
        try:
            ev:Ev = self.data_in.get_nowait()
            if ev.typ == Ev.Stop:
                return False
        except Empty:
            pass
        return True

    def proc_mic(self):
        try:
            orig_sr = self.mic_sampling_rate
            segsize = self._input_seg_size(orig_sr)
            self._update_butter( orig_sr )
            try:
                self.audioinput = sd.InputStream( device=int(self.mic_index), dtype=np.float32 )
            except:
                try:
                    self.audioinput.close()
                except:
                    pass
                self.audioinput = sd.InputStream( device=int(self.mic_index) )
            utc:float = time.time()
            self.audioinput.start()
            rz:int = 0
            self.proc_audio_filter( -1, np.zeros(segsize, dtype=np.float32), True, orig_sr )
            while self.audioinput is not None and self.proc_ctl():
                mute:bool = self.mic_mute
                seg2,overflow = self.audioinput.read( segsize )
                mute = mute or self.mic_mute
                seg1 = seg2[:,0]
                rz += len(seg1) 
                self.proc_audio_filter( utc, seg1, mute, orig_sr )
            self.proc_audio_filter( -2, np.zeros(segsize, dtype=np.float32), True, orig_sr )
        except:
            logger.exception('callback')
        finally:
            pass

    def proc_wav_file(self, *, wait:bool=False ):
        """waveファイルをリードして分割して次の処理に渡す"""
        self.state=1
        try:
            utc:float = 0.0
            with wave.open( self.source_file_path, 'rb' ) as stream:
                ch = stream.getnchannels()
                orig_sr = stream.getframerate()
                sw = stream.getsampwidth()
                total_length = stream.getnframes()
                segsize = self._input_seg_size(orig_sr)
                self._update_butter( orig_sr )
                total_time = total_length/orig_sr
                log_inverval = orig_sr * 5
                log_next = 0
                start_time = time.time()
                pos = 0
                audio_time:float = 0.0
                self.proc_audio_filter( -1, np.zeros(segsize, dtype=np.float32), True, orig_sr )
                mute:bool = False
                while self.proc_ctl():
                    x = stream.readframes(segsize)
                    if x is None or len(x)==0:
                        print( f"wave {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                        break
                    pcm2 = np.frombuffer( x, dtype=np.int16).reshape(-1,ch)
                    pcm = pad_to_length( pcm2[:,0], segsize )
                    audio_time = pos / orig_sr
                    if log_next<=pos:
                        log_next = pos + log_inverval
                        print( f"wave {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                    audio_f32 = pcm/32768.0
                    if wait:
                        wa = (start_time + audio_time) - time.time()
                        if wa>0:
                            time.sleep( wa )
                    self.proc_audio_filter( utc, audio_f32, mute, orig_sr )
                    pos += len(pcm)
                self.proc_audio_filter( -2, np.zeros(segsize, dtype=np.float32), True, orig_sr )
        except:
            logger.exception(f"filename:{self.source_file_path}")
        finally:
            self.state = 0

    def start_stt_data_file(self, *, wait:bool=False ):
        """SttDataから音声データをリードして分割して次の処理に渡す"""
        self.state=1
        try:
                stt_data:SttData = SttData.load( self.source_file_path )
                audio = stt_data['raw']
                if audio is None:
                    audio = stt_data['audio']
            
                utc:float = stt_data.utc
                orig_sr = stt_data.sample_rate
                segsize = self._input_seg_size(orig_sr)
                self._update_butter( orig_sr )
                total_length = len(audio)
                total_time = total_length/orig_sr
                log_inverval = orig_sr * 5
                log_next = 0
                start_time = time.time()
                self.proc_audio_filter( -1, np.zeros(segsize, dtype=np.float32), True, orig_sr )
                mute:bool = False
                for pos in range( 0, total_length, segsize ):
                    if not self.proc_ctl():
                        break
                    audio_f32 = pad_to_length( audio[pos:pos+segsize], segsize )
                    audio_time = pos/orig_sr
                    if log_next<=pos:
                        log_next=pos+log_inverval
                        print( f"SttData {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                    if wait:
                        wa = (start_time+audio_time) - time.time()
                        if wa>0.0:
                            time.sleep( wa )
                    self.proc_audio_filter( utc, audio_f32, mute, orig_sr )
                self.proc_audio_filter( -2, np.zeros(segsize, dtype=np.float32), True, orig_sr )
        except:
            logger.exception(f"filename:{self.source_file_path}")
        finally:
            self.state = 0

    def proc_debug_data(self, *, wait:bool=False ):
        """デバッグデータを分割して次の処理に渡す"""
        self.state=1
        try:
                utc:float = 0.0
                xx = 3
                orig_sr = int(self.sample_rate * xx)
                segsize = self._input_seg_size(orig_sr)
                self.sos=None
                total_length = int(orig_sr * 3 )
                print(f"[debug] sr:{orig_sr} segsize:{segsize} length:{total_length}")
                a = []
                j = 0
                s = 0
                xf = int( self.frame_size*xx)
                while len(a)<total_length:
                    if not self.proc_ctl():
                        break
                    a.append(s+1)
                    j += 1
                    if j>=xf:
                        s+=1
                        j=0
                audio:np.ndarray = np.array( a, dtype=np.float32 )
                total_time = total_length/orig_sr
                log_inverval = orig_sr * 5
                log_next = 0
                start_time = time.time()
                self.proc_audio_filter( -1, np.zeros(segsize, dtype=np.float32), True, orig_sr )
                mute:bool = False
                for pos in range( 0, total_length, segsize ):
                    audio_f32 = pad_to_length( audio[pos:pos+segsize], segsize )
                    audio_time = pos/orig_sr
                    if log_next<=pos:
                        log_next=pos+log_inverval
                        print( f"Debug {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                    if wait:
                        wa = (start_time+audio_time) - time.time()
                        if wa>0.0:
                            time.sleep( wa )
                    self.proc_audio_filter( utc, audio_f32, mute, orig_sr )
                self.proc_audio_filter( -2, np.zeros(segsize, dtype=np.float32), True, orig_sr )
        except:
            logger.exception(f"filename:{self.source_file_path}")
        finally:
            self.state = 0

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



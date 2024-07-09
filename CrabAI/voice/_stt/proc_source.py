import sys,os
import platform
from logging import getLogger
import time
import numpy as np
import pandas as pd
from threading import Thread
from multiprocessing.queues import Queue
from queue import Empty
import wave
import sounddevice as sd

from CrabAI.vmp import Ev, ShareParam
from .stt_data import SttData

logger = getLogger(__name__)

class Mic:
    def __init__(self,samplerate, device, dtype ):
        self.samplerate=samplerate
        self.device=device
        self.dtype = dtype
        self.InputStream:sd.InputStream = None

    def __enter__(self):
        # check parameters
        sd.check_input_settings( device=self.device, samplerate=self.samplerate, dtype=self.dtype )
        # channelsを指定したら内臓マイクで録音できないので指定してはいけない。
        self.InputStream = sd.InputStream( samplerate=self.samplerate, device=self.device )
        self.InputStream.start()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        # try:
        #     self.InputStream.abort(ignore_errors=True)
        # except:
        #     pass
        try:
            self.InputStream.stop(ignore_errors=True)
        except:
            pass
        try:
            self.InputStream.close(ignore_errors=True)
            return self.InputStream.__exit__(ex_type,ex_value,trace)
        except:
            pass
        finally:
            self.InputStream = None
        return True

    def read(self,sz):
        t = Thread( target=self._fn_timeout, daemon=True)
        t.start()
        return self.InputStream.read(sz)

    def _fn_timeout(self):
        try:
            time.sleep(0.5)
            self.__exit__(None,None,None)
        except:
            pass

def _mic_priority(x):
    devid = x['index']
    name = x['name']
    if 'default' in name:
        return 10000 + devid
    if 'USB' in name:
        return 20000 + devid
    return 90000 + devid

def get_mic_devices( *, samplerate=None, dtype=None, check:bool=True ):
    """マイクとして使えるデバイスをリストとして返す"""
    # 条件
    #sr:float = float(samplerate) if samplerate else 16000
    dtype = dtype if dtype else np.float32
    # select input devices
    inp_dev_list = [ x for x in sd.query_devices() if x['max_input_channels']>0 ]
    # select avaiable devices
    mic_dev_list = []
    for x in inp_dev_list:
        mid = x['index']
        sr = x.get('default_samplerate',float(samplerate))
        name = f"[{mid:2d}] {x['name']}"
        print(f"try get mic {name} {sr}")
        try:
            if check:
                lv=0
                with Mic( samplerate=sr, device=mid, dtype=dtype ) as audio_in:
                    frames,overflow = audio_in.read(1000)
                    if len(frames.shape)>1:
                        frames = frames[:,0]
                    lv = max(abs(frames))
                    if lv<1e-9:
                        print(f"NoSignal {name} {sr} {lv}")
                        logger.debug(f"NoSignal {name}")
                        #continue
                print(f"Avairable {name} {sr} {lv}")
                logger.debug(f"Avairable {name} {sr} {lv}")
            x['label'] = name
            mic_dev_list.append(x)
        except sd.PortAudioError as ex:
            logger.debug(f"NoSupport {name} {str(ex)}")
            #traceback.print_exc()
        except:
            logger.exception('mic')
            #traceback.print_exc()
    # sort
    mic_dev_list = sorted( mic_dev_list, key=_mic_priority)
    # for x in mic_dev_list:
    #     print(f"[{x['index']:2d}] {x['name']}")
    return mic_dev_list

def input_seg_size( sample_rate, orig_sr:int ) -> int:
    segsize = int( sample_rate * ( orig_sr/sample_rate ) * 0.1 )
    return segsize

def pad_to_length( arr, length):
    if len(arr) < length:
        pad_width = length - len(arr)
        padded_arr = np.pad(arr, (0, pad_width), mode='constant', constant_values=0)
        return padded_arr
    else:
        return arr

class SourceBase:

    StInit:int = 0
    StLoaded:int = 1
    StStarted:int = 2
    StAbort:int = 3
    StStopped:int = 4

    def __init__(self, conf:ShareParam, data_out:Queue, *, source, sampling_rate:int=None):
        self.state:int = SourceBase.StInit
        self.conf:ShareParam = ShareParam(conf)
        self.data_out:Queue = data_out
        self.source = source
        self.sampling_rate:int = int(sampling_rate) if isinstance(sampling_rate,(int,float)) and sampling_rate>0 else 16000
        self.orig_sr:int = 0
        self.seq:int = 0
        self.end_of_data:bool = False
        self.mute_sw:bool = False
        self.in_listen:bool = True
        self.in_talk:bool = False

    def load(self):
        if self.state == SourceBase.StInit:
            self.state = SourceBase.StLoaded
            self.orig_sr:int = 0
            self.seq:int = 0
            self.load_source()

    def start(self):
        self.load()
        if self.state == SourceBase.StLoaded:
            self.state = SourceBase.StStarted
            try:
                self.seq = 0
                self.end_of_data:bool = False
                self.start_source()
            except:
                logger.exception('callback')
                self.stop()

    def stop(self):
        x = self.state != SourceBase.StStopped
        self.state = SourceBase.StStopped
        try:
            if x:
                self.stop_source()
        except:
            pass
        try:
            self._put_end_of_data()
        except:
            pass

    def _put_data(self, utc:float, pos:int, audio:np.ndarray ):
        #print(f"_mic_callback {data.shape} {data.dtype}")
        raw:np.ndarray = audio
        if len(audio.shape)>1:
            raw = audio[:,0]
        #print(f"_mic_callback {raw.shape} {raw.dtype}")
        st = pos
        ed = pos + len(raw)
        stt_data:SttData = SttData( SttData.Audio, utc, st, ed, sample_rate=self.orig_sr, raw=raw, seq=self.seq)
        if self.mute_sw or not self.in_listen or self.in_talk:
            mute_array:np.ndarray = np.ones(10, dtype=np.float32)
            stt_data.hists = pd.DataFrame( {'mute': mute_array })
        self.mute_sw = not self.in_listen or self.in_talk
        self.data_out.put( stt_data )
        self.seq+=1

    def _put_end_of_data(self):
        if not self.end_of_data:
            self.end_of_data = True
            ev:Ev = Ev( self.seq, Ev.EndOfData )
            self.data_out.put( ev )

    def load_source(self):
        raise NotImplementedError()

    def start_source(self):
        raise NotImplementedError()

    def stop_source(self):
        raise NotImplementedError()

    def precheck_mute(self, *, in_talk=None, in_listen=None):
        pre_talk = self.in_talk
        pre_listen = self.in_listen
        before:bool = not pre_listen or pre_talk
        if isinstance(in_talk,bool) and pre_talk != in_talk:
            pre_talk = in_talk
        if isinstance(in_listen,bool) and pre_listen != in_listen:
            pre_listen = in_listen
        after:bool = not pre_listen or pre_talk
        return after, before

    def set_mute(self, *, in_talk=None, in_listen=None):
        before:bool = not self.in_listen or self.in_talk
        if isinstance(in_talk,bool) and self.in_talk != in_talk:
            # print(f"[STT] in_talk {self.in_talk} -> {in_talk}")
            self.in_talk = in_talk
        if isinstance(in_listen,bool) and self.in_listen != in_listen:
            # print(f"[STT] in_listen {self.in_listen} -> {in_listen}")
            self.in_listen = in_listen
        after:bool = not self.in_listen or self.in_talk
        # if after != before and not after:
        #     stack = traceback.extract_stack()
        #     filtered_stack = [frame for frame in stack if 'maeda/LLM/CrabVoice/' in frame.filename]
        #     stack_trace = ''.join(traceback.format_list(filtered_stack))
        #     logger.info(f"[STT] set_mute in_talk={in_talk} in_listen={in_listen}\n%s", stack_trace)
        return after, before

class MicSource(SourceBase):

    def __init__(self, conf:ShareParam, data_out:Queue, source, sampling_rate:int=None, mic_sampling_rate:int=None):
        super().__init__( conf, data_out, source=source, sampling_rate=sampling_rate )
        self.mic_sampling_rate = int(mic_sampling_rate) if isinstance(mic_sampling_rate,(int,float)) and mic_sampling_rate>self.sampling_rate else self.sampling_rate
        self.utc:float = time.time()
        self.pos:int = 0
        self.in_listen = False

    def load_source(self):
        if self.source is not None:
            inp_dev_list = get_mic_devices(samplerate=self.mic_sampling_rate, dtype=np.float32,check=False)
            if inp_dev_list and len(inp_dev_list)>0:
                self.source='default'
                if self.source=='' or self.source=='default':
                    self.source = inp_dev_list[0]['index']
                    self.mic_name = inp_dev_list[0]['name']
                    self.mic_sampling_rate = int( inp_dev_list[0].get('default_samplerate',self.mic_sampling_rate) )
                    logger.info(f"selected mic : {self.source} {self.mic_name} {self.mic_sampling_rate}")
                else:
                    for m in inp_dev_list:
                        if m['index'] == self.source:
                            self.mic_sampling_rate = int( m.get('default_samplerate',self.mic_sampling_rate) )
                            self.mic_name = m.get('name','mic')
                print(f"selected mic : {self.source} {self.mic_name} {self.mic_sampling_rate}")
            else:
                self.source = None

    def _mic_callback(self, data:np.ndarray, a,b,c ):
        xdata = data[:,0].copy()
        self._put_data( self.utc, self.pos, xdata )
        self.pos += len(xdata)

    def start_source(self):
        try:
            orig_sr = self.orig_sr = self.mic_sampling_rate
            segsize = input_seg_size( self.sampling_rate,orig_sr)
            self.audioinput = sd.InputStream( samplerate=orig_sr, blocksize=segsize, device=self.source, dtype=np.float32, callback=self._mic_callback )
            self.utc = time.time()
            self.pos = 0
            self.audioinput.start()
        except:
            logger.exception('callback')
        finally:
            pass

    def stop_source(self):
        try:
            self.audioinput.close()
        finally:
            self.audioinput = None

class ThreadSourceBase(SourceBase):
    def __init__(self, conf: ShareParam, data_out:Queue, *, source, sampling_rate:int ):
        super().__init__( conf, data_out, source=source, sampling_rate=sampling_rate )
        self.wait = False
        self.th:Thread = None

    def start_source(self):
        self.th = Thread( target=self._proc_thread, name='WavSource', daemon=True )
        self.th.start()

    def stop_source(self):
        self.th.join()

    def _proc_thread(self):
        raise NotImplementedError()

class WavSource(ThreadSourceBase):

    def load_source(self):
        try:
            with wave.open( self.source, 'rb' ) as stream:
                ch = stream.getnchannels()
                orig_sr = stream.getframerate()
                sw = stream.getsampwidth()
                total_length = stream.getnframes()
                total_time = total_length/orig_sr
                self.orig_sr = orig_sr
            print(f"load {self.source} {self.orig_sr:.0f}Hz {ch}Ch {total_time:.3f}(Sec) {total_length}(Samples)")
        except:
            logger.exception(f"filename:{self.source}")

    def _proc_thread(self):

        try:
            utc:float = 0.0
            with wave.open( self.source, 'rb' ) as stream:
                ch = stream.getnchannels()
                orig_sr = stream.getframerate()
                sw = stream.getsampwidth()
                total_length = stream.getnframes()
                segsize = input_seg_size( self.sampling_rate, orig_sr )
                total_time = total_length/orig_sr
                log_inverval = orig_sr * 5
                log_next = 0
                start_time = time.time()
                pos = 0
                audio_time:float = 0.0
                while self.state == SourceBase.StStarted:
                    x = stream.readframes(segsize)
                    if x is None or len(x)==0:
                        break
                    pcm2 = np.frombuffer( x, dtype=np.int16).reshape(-1,ch)
                    pcm = pad_to_length( pcm2[:,0], segsize )
                    audio_time = pos / orig_sr
                    if log_next<=pos:
                        log_next = pos + log_inverval
                        print( f"wave {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                    audio_f32 = pcm/32769.0
                    if self.wait:
                        wa = (start_time + audio_time) - time.time()
                        if wa>0:
                            time.sleep( wa )
                    self._put_data( utc, pos, audio_f32 )
                    pos += len(pcm)
                print( f"wave {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                self._put_end_of_data()
        except:
            logger.exception(f"filename:{self.source_file_path}")

class SttSource(ThreadSourceBase):

    def load_source(self):
        try:
            self.stt_data:SttData = SttData.load( self.source )
            orig_sr = self.orig_sr = self.stt_data.sample_rate
        except:
            logger.exception(f"filename:{self.source}")

    def _proc_thread(self):
        try:
                stt_data:SttData = self.stt_data
                audio = stt_data['raw']
                if audio is None:
                    audio = stt_data['audio']
            
                utc:float = stt_data.utc
                orig_sr = self.orig_sr = stt_data.sample_rate
                segsize = input_seg_size(orig_sr)
                total_length = len(audio)
                total_time = total_length/orig_sr
                log_inverval = orig_sr * 5
                log_next = 0
                start_time = time.time()
                for pos in range( 0, total_length, segsize ):
                    if self.state != SourceBase.StStarted:
                        break
                    audio_f32 = pad_to_length( audio[pos:pos+segsize], segsize )
                    audio_time = pos/orig_sr
                    if log_next<=pos:
                        log_next=pos+log_inverval
                        print( f"SttData {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                    if self.wait:
                        wa = (start_time+audio_time) - time.time()
                        if wa>0.0:
                            time.sleep( wa )
                    self._put_data( utc, pos, audio_f32 )
                print( f"SttData {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                self._put_end_of_data()
        except:
            logger.exception(f"filename:{self.source}")

class DbgSource(ThreadSourceBase):

    def load_source(self):
        pass

    def _proc_thread(self):
        """デバッグデータを分割して次の処理に渡す"""
        try:
                utc:float = 0.0
                xx = 3
                orig_sr = int(self.sampling_rate * xx)
                segsize = input_seg_size(orig_sr)
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
                for pos in range( 0, total_length, segsize ):
                    audio_f32 = pad_to_length( audio[pos:pos+segsize], segsize )
                    audio_time = pos/orig_sr
                    if log_next<=pos:
                        log_next=pos+log_inverval
                        print( f"Debug {audio_time:.2f}/{total_time:.2f} {pos}/{total_length}")
                    if self.wait:
                        wa = (start_time+audio_time) - time.time()
                        if wa>0.0:
                            time.sleep( wa )
                    self._put_data( utc, pos, audio_f32 )
                self._put_end_of_data()
        except:
            logger.exception(f"filename:{self.source_file_path}")


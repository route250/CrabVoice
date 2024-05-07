import sys
import os,platform
from logging import getLogger
import time
import traceback
from threading import Thread
import wave
import numpy as np
import librosa
import sounddevice as sd

logger = getLogger(__name__)

class Mic:
    def __init__(self,samplerate, device, dtype ):
        self.samplerate=samplerate
        self.device=device
        self.dtype = dtype
        self.InputStream=None

    def __enter__(self):
        # check parameters
        sd.check_input_settings( device=self.device, samplerate=self.samplerate, dtype=self.dtype )
        # channelsを指定したら内臓マイクで録音できないので指定してはいけない。
        self.InputStream = sd.InputStream( samplerate=self.samplerate, device=self.device )
        self.InputStream.start()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.InputStream.abort(ignore_errors=True)
        self.InputStream.stop(ignore_errors=True)
        self.InputStream.close(ignore_errors=True)

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

def get_mic_devices( *, samplerate=None, dtype=None ):
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
            with Mic( samplerate=sr, device=mid, dtype=dtype ) as audio_in:
                frames,overflow = audio_in.read(1000)
                if len(frames.shape)>1:
                    frames = frames[:,0]
                if max(abs(frames))<1e-9:
                    logger.debug(f"NoSignal {name}")
                    continue
            logger.debug(f"Avairable {name}")
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

class mic_to_audio:

    def __init__(self, *, sample_rate:int, callback ):
        self.state:int = 0
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000
        self.callback = callback
        self.mic_index = None
        self.mic_name = None
        self.mic_sampling_rate = None
        self.start_utc:float = 0
        self.audioinput:sd.InputStream = None

    def __getitem__(self,key):
        if 'mic_index'==key:
            return self.mic_index
        elif 'mic_name'==key:
            return self.mic_name
        return None

    def to_dict(self)->dict:
        keys = ['mic_index','mic_name']
        ret = {}
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        pass
        # if 'vad.mode'==key:
        #     if isinstance(val,(int,float)) and 0<=key<=3:
        #         self.vad_mode = int(key)

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def load(self,*,mic=None, sample_rate=None):
        self.sample_rate = sample_rate if isinstance(sample_rate,int) else self.sample_rate
        if not mic or mic==0 or mic=='' or mic=='default':
            inp_dev_list = get_mic_devices(samplerate=self.sample_rate, dtype=np.float32)
            if inp_dev_list and len(inp_dev_list)>0:
                self.mic_index = inp_dev_list[0]['index']
                self.mic_name = inp_dev_list[0]['name']
                self.mic_sampling_rate = int( inp_dev_list[0].get('default_samplerate',self.sample_rate) )
                logger.info(f"selected mic : {self.mic_index} {self.mic_name} {self.mic_sampling_rate}")
                print(f"selected mic : {self.mic_index} {self.mic_name} {self.mic_sampling_rate}")
            else:
                self.mic_index = None
        else:
            self.mic_index = mic

    def _fn_callback(self,indata: np.ndarray, frames: int, tm, status: sd.CallbackFlags):
        try:
            if self.start_utc<=0:
                self.start_utc = time.time()
            if self.mic_sampling_rate != self.sample_rate:
                audio_f32 = librosa.resample( indata, axis=0, orig_sr=self.mic_sampling_rate, target_sr=self.sample_rate )
            else:
                audio_f32 = indata
            self.callback( self.start_utc, audio_f32 )
        except:
            logger.exception('callback')

    def _fn_finished_callback(self):
        try:
            if self.start_utc<=0:
                self.start_utc = time.time()
            self.callback( self.start_utc, None )
        except:
            logger.exception('callback')

    def start(self):
        os_name = platform.system()
        if os_name == "Darwin":
            self.start_macos()
        else:
            self.start_linux()

    def start_linux(self):
        try:
            segsize = int( self.mic_sampling_rate*0.1 )
            self.audioinput = sd.InputStream( samplerate=self.mic_sampling_rate, blocksize=segsize, device=self.mic_index, dtype=np.float32, callback=self._fn_callback, finished_callback=self._fn_finished_callback )
            self.audioinput.start()
        except:
            logger.exception('callback')

    def start_macos(self):
        try:
            segsize = int( self.mic_sampling_rate*0.1 )
            def micinput():
                self.audioinput = sd.InputStream( samplerate=self.mic_sampling_rate, blocksize=segsize, device=self.mic_index, dtype=np.float32 )
                self.audioinput.start()
                try:
                    while self.audioinput is not None:
                        seg,overflow = self.audioinput.read( segsize )
                        self._fn_callback( seg, 1, 2, 3 )
                finally:
                    self._fn_finished_callback()
            tx = Thread( name='micinput', target=micinput, daemon=True )
            tx.start()
        except:
            logger.exception('callback')

    def set_pause(self,b):
        pass

    def stop(self):
        try:
            if self.audioinput is not None:
                self.audioinput.close()
                self.audioinput = None
        except:
            pass

    def __del__(self):
        try:
            if self.audioinput is not None:
                self.audioinput.close()
                self.audioinput = None
        except:
            pass

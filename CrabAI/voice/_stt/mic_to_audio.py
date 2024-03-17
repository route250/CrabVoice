import sys
import os
import logging
import time
import traceback
import wave
import numpy as np
import librosa
import sounddevice as sd

logger = logging.getLogger('mic_to_audio')

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
    sr:float = float(samplerate) if samplerate else 16000
    dtype = dtype if dtype else np.float32
    # select input devices
    inp_dev_list = [ x for x in sd.query_devices() if x['max_input_channels']>0 ]
    # select avaiable devices
    mic_dev_list = []
    for x in inp_dev_list:
        mid = x['index']
        name = f"[{mid:2d}] {x['name']}"
        try:
            # check parameters
            sd.check_input_settings( device=mid, samplerate=sr, dtype=dtype )
            # read audio data
            # channelsを指定したら内臓マイクで録音できないので指定してはいけない。
            with sd.InputStream( samplerate=sr, device=mid ) as audio_in:
                frames,overflow = audio_in.read(1000)
                audio_in.abort(ignore_errors=True)
                audio_in.stop(ignore_errors=True)
                audio_in.close(ignore_errors=True)
                if len(frames.shape)>1:
                    frames = frames[:,0]
                if max(abs(frames))<1e-9:
                    logger.debug(f"NoSignal {name}")
                    continue
            logger.debug(f"Avairable {name}")
            mic_dev_list.append(x)
        except sd.PortAudioError:
            logger.debug(f"NoSupport {name}")
        except:
            logger.exception('mic')
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
        self.device = None
        self.audioinput:sd.InputStream = None

    def load(self,*,mic=None, sample_rate=None):
        self.sample_rate = sample_rate if isinstance(sample_rate,int) else self.sample_rate
        if not mic or mic==0 or mic=='' or mic=='default':
            inp_dev_list = get_mic_devices(samplerate=self.sample_rate, dtype=np.float32)
            self.device = inp_dev_list[0]['index'] if inp_dev_list and len(inp_dev_list)>0 else None
        else:
            self.device = mic

    def start(self):
        try:
            segsize = int( self.sample_rate * ( self.sample_rate/self.sample_rate ) * 0.1 )
            self.audioinput = sd.InputStream( samplerate=self.sample_rate, blocksize=segsize, device=self.device, dtype=np.float32, callback=self.callback )
            self.audioinput.start()
        except:
            pass

    def set_pause(self,b):
        pass

    def stop(self):
        self.__del__()

    def __del__(self):
        try:
            if self.audioinput is not None:
                self.audioinput.close()
                self.audioinput = None
        except:
            pass

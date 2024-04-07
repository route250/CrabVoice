import sys
import os
import logging
import time
import traceback
import wave
import numpy as np
import librosa

from .stt_data import SttData

logger = logging.getLogger('wave_to_audio')

class wave_to_audio:

    def __init__(self, *, sample_rate:int, callback ):
        self.sample_rate = sample_rate
        self.callback = callback
        self.filename:str = None
        self.state:int = 0

    def __getitem__(self,key):
        if 'file'==key:
            return self.filename
        return None

    def to_dict(self)->dict:
        keys = ['file']
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

    def load(self, filename):
        try:
            librosa.resample( np.zeros( (1,1), dtype=np.float32), orig_sr=self.sample_rate*2, target_sr=self.sample_rate ) # preload of librosa
        except:
            traceback.print_exc()
        self.filename = filename

    def start(self, *, wait:bool=False ):
        if self.filename.endswith('.wav'):
            self.start_wav(wait=wait)
        elif self.filename.endswith('.npz'):
            self.start_stt_data(wait=wait)

    def start_wav(self, *, wait:bool=False ):
        self.state=1
        try:
            with wave.open( self.filename, 'rb' ) as stream:
                ch = stream.getnchannels()
                fr = stream.getframerate()
                sw = stream.getsampwidth()
                num_fr = stream.getnframes()
                segsize = int( self.sample_rate * ( fr/self.sample_rate ) * 0.1 )
                num_tm = num_fr/fr
                log_inverval = fr * 5
                i = 0
                l = 0
                start_time = time.time()
                frame_readed = 0
                while True:
                    x = stream.readframes(segsize)
                    if x is None or len(x)==0:
                        break
                    i+=segsize
                    pcm = np.frombuffer( x, dtype=np.int16).reshape(-1,ch)
                    frame_readed += len(pcm)
                    call_time = start_time + (frame_readed/fr)
                    l += len(pcm)
                    if l>log_inverval:
                        l=0
                        tm = i/fr
                        print( f"wave {tm:.2f}/{num_tm:.2f} {i}/{num_fr}")
                    orig_audio_f32 = pcm /32767.0
                    if fr != self.sample_rate:
                        audio_f32 = librosa.resample( orig_audio_f32, axis=0, orig_sr=fr, target_sr=self.sample_rate )
                    else:
                        audio_f32 = orig_audio_f32
                    wa = call_time - time.time()
                    if wait and wa>0:
                        time.sleep( wa )
                    self.callback( 0.0, audio_f32 )
                self.callback( 0.0, None )
        except:
            logger.exception(f"filename:{self.filename}")
        finally:
            self.state = 0

    def start_stt_data(self, *, wait:bool=False ):
        self.state=1
        try:
                stt_data:SttData = SttData.load( self.filename )
                stt_data.sample_rate
                audio = stt_data['raw']
                if audio is None:
                    audio = stt_data['audio']
            
                fr = stt_data.sample_rate
                num_fr = len(audio)
                segsize = int( self.sample_rate * ( fr/self.sample_rate ) * 0.1 )
                num_tm = num_fr/fr
                log_inverval = fr * 5
                logprints = 0
                start_time = time.time()
                for i in range( 0, num_fr, segsize ):
                    orig_audio_f32 = audio[i:i+segsize]
                    call_time = start_time + (i/fr)
                    logprints += len(orig_audio_f32)
                    if logprints>log_inverval:
                        logprints=0
                        tm = i/fr
                        print( f"wave {tm:.2f}/{num_tm:.2f} {i}/{num_fr}")

                    if fr != self.sample_rate:
                        audio_f32 = librosa.resample( orig_audio_f32, axis=0, orig_sr=fr, target_sr=self.sample_rate ).reshape(-1,1)
                    else:
                        audio_f32 = orig_audio_f32.reshape(-1,1)
                    wa = call_time - time.time()
                    if wait and wa>0:
                        time.sleep( wa )
                    self.callback( 0.0, audio_f32 )
                self.callback( 0.0, None )
        except:
            logger.exception(f"filename:{self.filename}")
        finally:
            self.state = 0

    def set_pause(self,b):
        pass

    def stop(self):
        if self.state==1:
            self.state=2

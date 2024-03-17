import sys
import os
import logging
import time
import traceback
import wave
import numpy as np
import librosa

logger = logging.getLogger('wave_to_audio')

class wave_to_audio:

    def __init__(self, *, sample_rate:int, callback ):
        self.sample_rate = sample_rate
        self.callback = callback
        self.filename:str = None
        self.state:int = 0

    def load(self, filename):
        try:
            librosa.resample( np.zeros( (1,1), dtype=np.float32), orig_sr=self.sample_rate*2, target_sr=self.sample_rate ) # preload of librosa
        except:
            traceback.print_exc()
        self.filename = filename

    def start(self, *, wait:bool=False ):
        self.state=1
        try:
            with wave.open( self.filename, 'rb' ) as stream:
                ch = stream.getnchannels()
                fr = stream.getframerate()
                sw = stream.getsampwidth()
                num_fr = stream.getnframes()
                segsize = int( self.sample_rate * ( fr/self.sample_rate ) * 0.1 )
                num_tm = num_fr/fr
                wait_time = segsize/fr
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
                    self.callback( audio_f32 )
        except:
            logger.exception(f"filename:{self.filename}")
        finally:
            self.state = 0

    def set_pause(self,b):
        pass

    def stop(self):
        if self.state==1:
            self.state=2

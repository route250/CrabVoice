import sys
import os
import time
import wave
import numpy as np
from scipy import signal

from faster_whisper import WhisperModel

model_size = "large-v3"
audio_file='nakagawke01.wav'

def test1():
    with wave.open(audio_file,'rb') as stream:
        ch=stream.getnchannels()
        fr=stream.getframerate()
        print( f"wave ch:{ch} rate:{fr}" )
        sz=fr*30//1000 # 30ms
        szz= sz*ch
        while True:
            fb:bytes=stream.readframes(szz)
            if fb is None or len(fb)==0:
                break
            audiodata:np.ndarray=np.frombuffer(fb,dtype=np.int16).reshape(-1,ch)
        print(f"shape {audiodata.shape}")
                      


def test2():
    print( f"#Load model")
    t0=time.time()
    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16" )
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    t1=time.time()
    print( f"   time:{t1-t0}(sec)")

    print( f"#Start")

    fr16k:int = 16000
    with wave.open(audio_file,'rb') as stream:
        ch=stream.getnchannels()
        fr=stream.getframerate()
        print( f"wave ch:{ch} rate:{fr}" )
        sz=fr*30//1000 # 30ms
        szz= sz*ch
        while True:
            fb:bytes=stream.readframes(szz)
            if fb is None or len(fb)==0:
                break
            audio2_i16:np.ndarray=np.frombuffer(fb,dtype=np.int16).reshape(-1,ch)
            audio_i16 = audio2_i16[:,0]
            audio_f32:np.ndarray = audio_i16.astype(np.float32)/32767.0
            target_num = (len(audio_f32)* fr16k) // fr16k
            audio_16k = signal.resample( audio_f32, num=target_num )
            segments, info = model.transcribe( audio_16k, language='ja')
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    t2=time.time()
    print( f"# TIME {t1-t0} {t2-t1}")

if __name__ == "__main__":
    test2()
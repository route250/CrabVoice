import sys,os
sys.path.append(os.getcwd())
import platform
from logging import getLogger
import time
import numpy as np
from multiprocessing import Queue as PQ
from queue import Empty
import wave
import sounddevice as sd
import librosa
from scipy import signal

from CrabAI.voice._stt.stt_data import SttData
from CrabAI.vmp import Ev, VFunction, VProcessGrp
from CrabAI.voice._stt.proc_source import MicSource, WavSource, SttSource, DbgSource
from CrabAI.voice._stt.proc_source_to_audio import SourceToAudio, shrink

from CrabAI.voice._stt.proc_stt_engine import SttEngine

def test001():

    ctl_out = PQ()
    data_in = PQ()
    data_out= PQ()
    # 'testData/voice_mosimosi.wav'
    th = VProcessGrp( SourceToAudio, 1, data_in, data_out, ctl_out, mic=None, source=None, sample_rate=16000 )

    th.start()

    while True:
        try:
            stt_data:SttData = data_out.get( timeout=0.1 )
            print( f"[OUT] {stt_data} audio:{stt_data.audio.shape} {stt_data.hists.shape}")
            al:int = stt_data.audio.shape[0]
            hl:int = stt_data.hists.shape[0]
            seglen:int = int( al/hl )
            if int(al/seglen) != hl:
                print(f"ERROR?")
            for s in range(0,al,seglen):
                first_value = stt_data.raw[s]
                last_value = stt_data.raw[s+seglen-1]
                seg = stt_data.raw[s:s+seglen]
                seg_max = np.max(seg)
                seg_min = np.min(seg)
                hi = stt_data.hists['hi'][int(s/seglen)]
                lo = stt_data.hists['lo'][int(s/seglen)]
                print( f"    No.{int(s/seglen)} Value:{first_value:.3f}:{last_value:.3f} hi:{seg_max:.3f}/{hi:.3f} lo:{seg_min:.3f}/{lo:.3f}")
        except:
            if th.is_alive():
                continue
            else:
                break

    th.join()

def test002():
    data = np.array( [i for i in range(1,11)], dtype=np.float32 )
    data2 = shrink( data, 6 )

    print( data )
    print( data2 )

def test003():

    print("[TEST003] Test start")
    # 
    wav_file = 'testData/voice_mosimosi.wav'
    wav_file = 'testData/voice_command.wav'
    wav_file = 'testData/audio_100Km_5500rpm.wav'
    # wav_file = 'testData/nakagawke01.wav'
    #src = MicSource( data_in1, mic_index=10, sampling_rate=16000 )
    blk:SttEngine = SttEngine( source=wav_file, sample_rate=16000, num_vosk=3 )

    print("[TEST003] Process start")
    blk.start()

    print("[TEST003] Loop")
    st=time.time()
    while True:
        if (time.time()-st)>180.0:
            blk.stop()
        try:
            stt_data:SttData = blk.get_ctl( timeout=0.1 )
            print( f"[CTL] {stt_data}")
        except Empty:
            pass
        try:
            stt_data:SttData = blk.get_data( timeout=0.1 )
            print( f"[OUT] {stt_data}")
        except Empty:
            pass
        if blk.is_alive():
            continue
        else:
            break
    blk.join()


def test_mic():
    q=PQ()
    src=MicSource(q,10,16000)
    src.start()
    time.sleep(10)
    src.stop()

if __name__ == "__main__":
    test003()

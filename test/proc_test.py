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
from CrabAI.voice._stt.proc_source_to_audio import SourceToAudio, shrink
from CrabAI.voice._stt.proc_audio_to_segment import AudioToSegment
from CrabAI.voice._stt.proc_segment_to_voice import SegmentToVoice
from CrabAI.voice._stt.proc_voice_to_text import VoiceToText

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
    ctl_out = PQ()
    data_in1 = PQ()
    data_in2 = PQ()
    data_in3 = PQ()
    data_in4 = PQ()
    data_out= PQ()
    # 
    wav_file = 'testData/voice_mosimosi.wav'
    wav_file = 'testData/voice_command.wav'
    #wav_file = 'testData/audio_100Km_5500rpm.wav'
    wav_file = 'testData/nakagawke01.wav'
    th1 = VProcessGrp( SourceToAudio, 1, data_in1, data_in2, ctl_out, source=wav_file, sample_rate=16000 )
    th2 = VProcessGrp( AudioToSegment, 1, data_in2, data_in3, ctl_out, sample_rate=16000 )
    th3 = VProcessGrp( SegmentToVoice, 3, data_in3, data_in4, ctl_out, sample_rate=16000 )
    th4 = VProcessGrp( VoiceToText, 1, data_in4, data_out, ctl_out )

    print("[TEST003] Process start")
    th4.start()
    th3.start()
    th2.start()
    th1.start()

    print("[TEST003] Loop")
    while True:
        try:
            stt_data:SttData = ctl_out.get( timeout=0.1 )
            print( f"[CTL] {stt_data}")
        except Empty:
            pass
        try:
            stt_data:SttData = data_out.get( timeout=0.1 )
            print( f"[OUT] {stt_data}")
        except Empty:
            pass
        if th1.is_alive() or th2.is_alive() or th3.is_alive() or th4.is_alive():
            continue
        else:
            break

    th1.join()
    print("[TEST003] th1 exit")
    th2.join()
    print("[TEST003] th2 exit")
    th3.join()
    print("[TEST003] th3 exit")
    th4.join()
    print("[TEST003] th4 for exit")

if __name__ == "__main__":
    test003()

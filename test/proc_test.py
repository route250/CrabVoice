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

class SttBlock:
    def __init__(self, source, *, sample_rate:int=16000, num_vosk:int=2 ):
        self.sample_rate:int = sample_rate if isinstance(sample_rate,(int,float)) and sample_rate>16000 else 16000

        ctl_out = self.ctl_out = PQ()
        data_in1 = self.data_in1 = PQ()

        if isinstance(source,str):
            if source.endswith('.wav'):
                self.src = WavSource( data_in1, ctl_out, sampling_rate=self.sample_rate, source=source  )
            elif source.endswith('.pyz'):
                self.src = SttSource( data_in1, ctl_out, sampling_rate=self.sample_rate, source=source  )
            else:
                raise ValueError(f'invalid source: {source}')
        elif isinstance(source,int):
            self.src = MicSource( data_in1, ctl_out, sampling_rate=self.sample_rate, source=source  )
        else:
            raise ValueError(f'invalid source: {source}')
        
        data_in2 = self.data_in2 = PQ()
        data_in3 = self.data_in3 = PQ()
        data_in4 = self.data_in4 = PQ()
        data_out = self.data_out= PQ()

        self.th1 = VProcessGrp( SourceToAudio, 1, data_in1, data_in2, ctl_out, sample_rate=self.sample_rate )
        self.th2 = VProcessGrp( AudioToSegment, 1, data_in2, data_in3, ctl_out, sample_rate=self.sample_rate )
        self.th3 = VProcessGrp( SegmentToVoice, num_vosk, data_in3, data_in4, ctl_out )
        self.th4 = VProcessGrp( VoiceToText, 1, data_in4, data_out, ctl_out )

    def start(self):
        self.th4.start()
        self.th3.start()
        self.th2.start()
        self.th1.start()
        self.src.start()

    def stop(self):
        self.src.stop()

    def is_alive(self):
        if self.th1.is_alive() or self.th2.is_alive() or self.th3.is_alive() or self.th4.is_alive():
            return True
        else:
            return False

    def join(self):
        self.th1.join()
        print("[TEST003] th1 exit")
        self.th2.join()
        print("[TEST003] th2 exit")
        self.th3.join()
        print("[TEST003] th3 exit")
        self.th4.join()
        print("[TEST003] th4 for exit")        

    def get_data(self, timeout:float=None ):
        stt_data:SttData = self.data_out.get( timeout=0.1 )
        return stt_data
    def get_ctl(self, timeout:float=None ):
        stt_data:SttData = self.ctl_out.get( timeout=0.1 )
        return stt_data

def test003():

    print("[TEST003] Test start")
    # 
    wav_file = 'testData/voice_mosimosi.wav'
    wav_file = 'testData/voice_command.wav'
    wav_file = 'testData/audio_100Km_5500rpm.wav'
    # wav_file = 'testData/nakagawke01.wav'
    #src = MicSource( data_in1, mic_index=10, sampling_rate=16000 )
    blk:SttBlock = SttBlock( source=wav_file, sample_rate=16000, num_vosk=3 )

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
            stt_data:SttData = blk.get_( timeout=0.1 )
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

import sys,os
import platform
from logging import getLogger
import time
import numpy as np
from threading import Thread
from multiprocessing import Queue as PQ
from queue import Empty

from CrabAI.voice._stt.stt_data import SttData
from CrabAI.vmp import Ev, VFunction, VProcessGrp
from CrabAI.voice._stt.proc_source import MicSource, WavSource, SttSource, DbgSource
from CrabAI.voice._stt.proc_source_to_audio import SourceToAudio, shrink
from CrabAI.voice._stt.proc_audio_to_segment import AudioToSegment
from CrabAI.voice._stt.proc_segment_to_voice import SegmentToVoice
from CrabAI.voice._stt.proc_voice_to_text import VoiceToText

def safe_is_alive( t ):
    try:
        return t.is_alive()
    except:
        return False

def safe_join( t ):
    try:
        return t.join()
    except:
        return False

class SttEngine:
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

        self.prc1 = VProcessGrp( SourceToAudio, 1, data_in1, data_in2, ctl_out, sample_rate=self.sample_rate )
        self.prc2 = VProcessGrp( AudioToSegment, 1, data_in2, data_in3, ctl_out, sample_rate=self.sample_rate )
        self.prc3 = VProcessGrp( SegmentToVoice, num_vosk, data_in3, data_in4, ctl_out )
        self.prc4 = VProcessGrp( VoiceToText, 1, data_in4, data_out, ctl_out )
        self.th:Thread = None

    def load(self):
        pass
    
    def _th_loop(self, callback):

        if callback is None:
            return

        while True:
            try:
                stt_data:SttData = self.get_ctl( timeout=0.1 )
                q1=True
                if stt_data.typ == SttData.Dump:
                    print( f"[DUMP] {stt_data}")
            except Empty:
                q1=False

            try:
                stt_data:SttData = self.get_data( timeout=0.1 )
                q2=True
                print( f"[OUT] {stt_data}")
                if stt_data.typ == SttData.Text:
                    callback( stt_data )
            except Empty:
                q2=False

            if self.SttEngine.is_alive():
                continue
            else:
                break


    def start(self, callback=None):
        self.prc4.start()
        self.prc3.start()
        self.prc2.start()
        self.prc1.start()
        self.src.start()

        if callback is not None:
            self.th:Thread = Thread( target=self._th_loop, args=(callback,), daemon=True )
            self.th.start()

    def stop(self):
        self.src.stop()
                # self.data_in1.put( Ev(0, Ev.Stop) )
                # self.data_in2.put( Ev(0, Ev.Stop) )
                # self.data_in3.put( Ev(0, Ev.Stop) )
                # self.data_in3.put( Ev(0, Ev.Stop) )
                # self.data_in4.put( Ev(0, Ev.Stop) )

    def is_alive(self):
        if safe_is_alive(self.prc1) or safe_is_alive(self.prc2) or safe_is_alive(self.prc3) or safe_is_alive(self.prc4) or safe_is_alive(self.th):
            return True
        else:
            return False

    def join(self):
        safe_join(self.prc1)
        print("[TEST003] th1 exit")
        safe_join(self.prc2)
        print("[TEST003] th2 exit")
        safe_join(self.prc3)
        print("[TEST003] th3 exit")
        safe_join(self.prc4)
        print("[TEST003] th4 for exit")        
        safe_join(self.th)

    def get_data(self, timeout:float=None ):
        stt_data:SttData = self.data_out.get( timeout=0.1 )
        return stt_data

    def get_ctl(self, timeout:float=None ):
        stt_data:SttData = self.ctl_out.get( timeout=0.1 )
        return stt_data


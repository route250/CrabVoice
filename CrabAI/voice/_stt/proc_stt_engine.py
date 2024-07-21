import sys,os
import platform
from logging import getLogger
import time
import numpy as np
from threading import Thread
from multiprocessing import Queue as PQ
from multiprocessing import Array
from queue import Empty

from CrabAI.voice._stt.stt_data import SttData
from CrabAI.vmp import Ev, ShareParam, VFunction, VProcessGrp
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

def precheck( a:bool, b:bool ) -> bool:
    return a if isinstance(a,bool) else b

class SttEngine(ShareParam):

    @staticmethod
    def get_defult_audio_butter():
        return SourceToAudio.DEFAULT_BUTTER

    @staticmethod
    def get_defult_segment_butter():
        return SegmentToVoice.DEFAULT_BUTTER

    @staticmethod
    def load_default( conf:ShareParam ):
        SourceToAudio.load_default( conf )
        AudioToSegment.load_default( conf )
        SegmentToVoice.load_default( conf )
        VoiceToText.load_default( conf )

    def __init__(self, source, *, conf:ShareParam=None, sample_rate:int=16000, num_vosk:int=2 ):
        super().__init__( conf )
        self.sample_rate:int = sample_rate if isinstance(sample_rate,(int,float)) and sample_rate>16000 else 16000
        self.started:bool = False
        dump_out = self.dump_out = PQ()
        data_in1 = self.data_in1 = PQ()
        if not isinstance(conf,ShareParam):
            SttEngine.load_default(self)
        if isinstance(source,str):
            if source.endswith('.wav'):
                self.src = WavSource( self._share_array, data_in1, sampling_rate=self.sample_rate, source=source  )
            elif source.endswith('.pyz'):
                self.src = SttSource( self._share_array, data_in1, sampling_rate=self.sample_rate, source=source  )
            else:
                raise ValueError(f'invalid source: {source}')
        elif isinstance(source,int):
            self.src = MicSource( self._share_array, data_in1, sampling_rate=self.sample_rate, source=source  )
        else:
            raise ValueError(f'invalid source: {source}')
        
        data_in2 = self.data_in2 = PQ()
        data_in3 = self.data_in3 = PQ()
        data_in4 = self.data_in4 = PQ()
        data_out = self.data_out= PQ()

        self.prc1 = VProcessGrp( SourceToAudio, 1, self._share_array, data_in1, data_in2, sample_rate=self.sample_rate )
        self.prc2 = VProcessGrp( AudioToSegment, 1, self._share_array, data_in2, data_in3, dump_out, sample_rate=self.sample_rate )
        self.prc3 = VProcessGrp( SegmentToVoice, num_vosk, self._share_array, data_in3, data_in4 )
        self.prc4 = VProcessGrp( VoiceToText, 1, self._share_array, data_in4, data_out )
        self.th:Thread = None

    def load(self):
        pass
    
    def start(self):
        self.prc4.start()
        self.prc3.start()
        self.prc2.start()
        self.prc1.start()
        self.src.set_mute(True)
        self.src.start()

    def configure(self, **kwargs):
        e:Ev = Ev( 0, Ev.Config, **kwargs )
        self.data_in1.put( e )
        self.data_in2.put( e )
        self.data_in3.put( e )
        self.data_in4.put( e )

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
        if not self.dump_out.empty():
            try:
                stt_data:SttData = self.dump_out.get_nowait()
                stt_data = self._mute_detect(stt_data)
                return stt_data
            except Empty:
                pass
        stt_data:SttData = self.data_out.get( timeout=0.1 )
        stt_data = self._mute_detect(stt_data)
        return stt_data

    def _mute_detect(self,stt_data:SttData|None):
        if stt_data is None:
            return stt_data
        if stt_data.typ == Ev.MuteOn or stt_data.typ == Ev.MuteOff:
            # muteの情報からSTTモジュールがスタートしたか判定
            if self.started:
                raise Empty() # 開始隅ならmute情報は無視する
            self.started = True
            print(f"[STT]Started {stt_data}")
        return stt_data

    def tick_time(self, time_sec:float ):
        pass

    def get_mute(self, b ):
        if isinstance(self.src,MicSource):
            return self.src.get_mute()
        return False

    def set_mute(self, b ):
        if isinstance(self.src,MicSource):
            return self.src.set_mute(b)
        return False,False

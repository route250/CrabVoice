import sys,os
import platform
from logging import getLogger
import numpy as np
from multiprocessing import Queue
from queue import Empty
import time
from urllib.error import URLError, HTTPError
from logging import getLogger
logger = getLogger(__name__)

from CrabAI.vmp import Ev, VFunction, VProcess
from .stt_data import SttData
from .recognizer_google import RecognizerGoogle

class VoiceToText(VFunction):
    def __init__(self, data_in:Queue, data_out:Queue, ctl_out:Queue ):
        super().__init__(data_in,data_out,ctl_out)
        self.model='google'
        self.speech_state=0

    def load(self):
        pass

    def proc(self, ev ):
        if isinstance(ev,SttData):
            if SttData.Voice==ev.typ or SttData.PreVoice == ev.typ:
                self.proc_voice(ev)
            else:
                self.output_ev(ev)
        else:
            self.output_ev(ev)

    def proc_voice(self,stt_data:SttData):
        try:
            if self.speech_state!=2:
                self.speech_state=2
                self.proc_output_event( SttData( SttData.Start, stt_data.utc, stt_data.start, stt_data.start, stt_data.sample_rate, seq=stt_data.seq) )
            audio = stt_data.audio
            next_typ = SttData.Text if SttData.Voice == stt_data.typ else SttData.PreText
            if len(audio)>0:
                # 音量調整
                peek = np.max(audio)
                if peek<0.8:
                    audio = audio * (0.8/peek)
                t0 = time.time()
                try:
                    text, confidence = RecognizerGoogle.recognize( audio, sample_rate=16000 )
                except (HTTPError,URLError,TimeoutError) as ex:
                    logger.error( f"recognize {self.model} {str(ex)}")
                    next_typ = SttData.NetErr
                t1 = time.time()
                logger.debug( f"recognize {self.model} time {t1-t0:.4f}/{len(audio)/stt_data.sample_rate:.4f}(sec)")
            else:
                text = ''
            stt_data.typ = next_typ
            stt_data.content = text
            self.proc_output_event(stt_data)
        except:
            logger.exception("audio to text")
        finally:
            logger.info("exit audio to text")
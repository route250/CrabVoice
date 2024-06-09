import sys,os
import platform
from logging import getLogger
import numpy as np
from multiprocessing import Queue
import time
from urllib.error import URLError, HTTPError
from logging import getLogger
logger = getLogger(__name__)

from CrabAI.vmp import Ev, ShareParam, VFunction, VProcess
from .stt_data import SttData
from .recognizer_google import RecognizerGoogle

class VoiceToText(VFunction):

    @staticmethod
    def load_default( conf:ShareParam ):
        if isinstance(conf,ShareParam):
            pass

    def __init__(self, proc_no:int, num_proc:int, conf:ShareParam, data_in:Queue, data_out:Queue ):
        super().__init__(proc_no,num_proc,conf,data_in,data_out)
        self.model='google'
        self.speech_state=0

    def load(self):
        pass

    def reload_share_param(self):
        return

    def proc(self, ev ):
        if isinstance(ev,SttData):
            if SttData.Voice==ev.typ or SttData.PreVoice == ev.typ:
                self.proc_voice(ev)
            else:
                self.proc_output_event(ev)
        else:
            if ev.typ==Ev.EndOfData:
                pass
            else:
                self.proc_output_event(ev)

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
                    if text is None:
                        next_typ = Ev.Nop
                except (HTTPError,URLError,TimeoutError) as ex:
                    logger.error( f"recognize {self.model} {str(ex)}")
                    text = None
                    next_typ = SttData.NetErr
                t1 = time.time()
                logger.debug( f"recognize {self.model} time {t1-t0:.4f}/{len(audio)/stt_data.sample_rate:.4f}(sec)")
            else:
                text = ''
            if next_typ != Ev.Nop:
                stt_data.typ = next_typ
                stt_data.content = text
                self.proc_output_event(stt_data)
        except:
            logger.exception("audio to text")
        finally:
            logger.info("exit audio to text")
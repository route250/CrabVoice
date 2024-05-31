import sys,os,traceback
import time
from threading import Thread, Condition
from queue import Empty

import numpy as np

#from ..stt import RecognizerGoogle, VoiceSplitter, SttEngine
from .stt import SttData
from CrabAI.voice._stt.proc_source import get_mic_devices
from CrabAI.voice._stt.proc_stt_engine import SttEngine

from .tts import TtsEngine

import logging
logger = logging.getLogger(__name__)

class VoiceTalkEngine:
    EOT:str = TtsEngine.EOT
    """
    音声会話のためのエンジン。マイクから音声認識と、音声合成を行う
    """
    ST_STOPPED:int = 0
    ST_TALK:int = 10
    ST_TALK_END:int = 11
    ST_LISTEN:int = 20
    ST_LISTEN_END: int = 21

    def __init__(self, *, speaker:int=46, record_samplerate:int=1600 ):
        self._status = VoiceTalkEngine.ST_STOPPED
        self._callback = None
        self.text_lock:Condition = Condition()
        self.text_buf=[]
        self.text_confidence = 1.0
        self.text_stat=0
        self.th:Thread = None
        self.stt:SttEngine = None
        self.tts:TtsEngine = TtsEngine( speaker=speaker, talk_callback=self._tts_callback)

    def __getitem__(self,key):
        if self.stt is not None:
            val = self.stt[key]
            if val is not None:
                return val
        if self.tts is not None:
            val = self.tts[key]
            if val is not None:
                return val
        return None

    def to_dict(self)->dict:
        ret = self.stt.to_dict() if self.stt is not None else {}
        if self.tts is not None:
            ret.update( self.tts.to_dict() )
        return ret

    def __setitem__(self,key,val):
        if self.stt is not None:
            self.stt[key]=val
        if self.tts is not None:
            self.tts[key] = val

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def _fn_callback(self, stat:int, *, listen_text=None, confidence=None, talk_text=None, talk_emotion=None, talk_model=None ):
        if stat == VoiceTalkEngine.ST_LISTEN:
            self.tts.play_listn_in()
        elif stat == VoiceTalkEngine.ST_LISTEN_END:
            self.tts.play_listen_out()

        if self._callback is not None:
            try:
                self._callback( stat, listen_text=listen_text, confidence=None, talk_text=talk_text, talk_emotion=talk_emotion, talk_model=talk_model )
            except:
                logger.exception('')
        else:
            if stat == VoiceTalkEngine.ST_LISTEN:
                logger.info( f"[VoiceTalkEngine] listen {listen_text} {confidence}" )
            elif stat == VoiceTalkEngine.ST_LISTEN_END:
                logger.info( f"[VoiceTalkEngine] listen {listen_text} {confidence} __EOT__" )
            elif stat == VoiceTalkEngine.ST_TALK:
                logger.info( f"[VoiceTalkEngine] talk {talk_text}" )
            elif stat == VoiceTalkEngine.ST_TALK_END:
                logger.info( f"[VoiceTalkEngine] talk END" )

    def load(self, *, stt=True):
        if stt:
            mic_list = get_mic_devices(samplerate=16000, dtype=np.float32)
            if len(mic_list)>0:
                src = mic_list[0]['index']
                self.stt = SttEngine( source=src, sample_rate=16000 )
                self.stt.load()

    def _th_loop(self):
        try:
            self.stt.start()
            while self._status != VoiceTalkEngine.ST_STOPPED:
                try:
                    stt_data:SttData = self.stt.get_ctl( timeout=0.1 )
                    q1=True
                    if stt_data.typ == SttData.Dump:
                        print( f"[DUMP] {stt_data}")
                except Empty:
                    q1=False

                try:
                    stt_data:SttData = self.stt.get_data( timeout=0.1 )
                    q2=True
                    print( f"[OUT] {stt_data}")
                    if stt_data.typ == SttData.Text or stt_data.typ == SttData.Term:
                        self._fn_stt_callback( stt_data )
                except Empty:
                    q2=False

                if self.stt.is_alive():
                    continue
                else:
                    break

        except:
            traceback.print_exc()
            pass
        finally:
            self.stt.stop()
            self.stt.join()

    def start(self, *, stt=True):
        self._status = VoiceTalkEngine.ST_LISTEN
        if stt and self.stt is not None:
            self.th:Thread = Thread( target=self._th_loop, daemon=True )
            self.th.start()

    def stop(self):
        self._status = VoiceTalkEngine.ST_STOPPED
        if self.th is not None:
            self.th.join
    
    def tick_time(self, time_sec:float):
        self.tts.tick_time(time_sec)
        self.stt.tick_time(time_sec)

    def get_recognized_text(self):
        exit_time = time.time()+2.0
        with self.text_lock:
            while time.time()<exit_time:
                if self.text_stat==3:
                    self.text_stat = 0
                    if len(self.text_buf)>0:
                        text = ' '.join(self.text_buf)
                        self.text_buf = []
                        confs = self.text_confidence
                        return text, confs
                else:
                    self.text_lock.wait(0.5)
        return None,None

    def _fn_stt_callback(self, stt_data:SttData ):
        start_sec = stt_data.start/stt_data.sample_rate
        end_sec = stt_data.end/stt_data.sample_rate
        typ:int = stt_data.typ
        stat:str = SttData.type_to_str(typ)
        texts = stt_data.content
        confidence:float = 1.0
        copy_texts = []
        copy_confidence = 1.0
        s = -1
        if SttData.Start==typ:
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} START")
            s = VoiceTalkEngine.ST_LISTEN
            with self.text_lock:
                if texts is not None and len(texts)>0:
                    self.text_buf.append(texts)
                self.text_stat = 1
            copy_confidence = 1.0
        elif SttData.Text==typ:
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} {texts} {confidence}")
            s = VoiceTalkEngine.ST_LISTEN
            with self.text_lock:
                if texts is not None and len(texts)>0:
                    self.text_buf.append(texts)
                copy_texts = [ t for t in self.text_buf]
                self.text_stat = 2
            copy_confidence = self.text_confidence = confidence
        elif SttData.Term==typ:
            with self.text_lock:
                if texts is not None and len(texts)>0:
                    self.text_buf.append(texts)
                copy_texts = [ t for t in self.text_buf]
                if self.text_stat == 3 or self.text_stat == 0:
                    return
                self.text_stat = 3
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} {texts} {confidence} EOT")
            s = VoiceTalkEngine.ST_LISTEN_END
            copy_confidence = self.text_confidence = confidence
        elif SttData.Dump==typ:
            return
        elif SttData.NetErr==typ:
            self.tts.play_error1()
            return
        else:
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} {texts} {confidence} EOT")
            return
        self._fn_callback( s, listen_text=copy_texts, confidence=copy_confidence )

    def _tts_callback(self, text:str, emotion:int, model:str):
        """音声合成からの通知により、再生中は音声認識を止める"""
        if text:
            logger.info( f"[TTS] {text}")
            self._status = VoiceTalkEngine.ST_TALK
            self.stt.set_pause( True )
            self._fn_callback( VoiceTalkEngine.ST_TALK, talk_text=text, talk_emotion=emotion, talk_model=model )
        else:
            logger.info( f"[TTS] stop")
            self._status = VoiceTalkEngine.ST_LISTEN
            self._fn_callback( VoiceTalkEngine.ST_TALK_END, talk_text=None )
            self.stt.set_pause( False )

    def add_talk(self, text ):
        if self.stt is not None:
            self.stt.set_pause( True )
        if self.tts is not None:
            self.tts.add_talk( text )
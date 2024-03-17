from threading import Thread, Condition
from queue import Queue, PriorityQueue, Empty
import time
import logging
import json
from heapq import heapify, heappop, heappush

import numpy as np
import vosk
from vosk import KaldiRecognizer

from .stt_data import SttData
from . import recognizer_vosk
from .audio_to_segment import AudioToSegment

logger = logging.getLogger("AudioToVoide")

class AudioToVoice:
    def __init__(self, *, callback, sample_rate:int=16000, wave_dir=None):
        self.sample_rate = sample_rate if isinstance(sample_rate,int) else 16000
        self._state:int = 0
        self._lock:Condition = Condition()
        self._queue:Queue = Queue()
        self.use_vosk:int = 2
        self._thread:list[Thread] = [None] * self.use_vosk
        self.callback = callback
        self.audio_to_segment:AudioToSegment = AudioToSegment( callback=self._fn_callback, wave_dir=wave_dir )
        self.vosk: list[KaldiRecognizer] = [None] * len(self._thread)
        self.vosk_model: vosk.Model = None
        self.vosk_spk: vosk.SpkModel = None
        self.vosk_max_len:int = int(self.sample_rate*1.7)
        # 
        self.input_count:int = 0
        self.output_count:int = 0
        self.output_queue:list = []
        heapify(self.output_queue)

    def load(self):
        self.audio_to_segment.load()
        if self.use_vosk>0 and self.vosk[0] is None:
            if self.vosk_model is None:
                logger.info( f"load vosk lang model" )
                self.vosk_model = recognizer_vosk.get_vosk_model(lang="ja")
            # if self.vosk_spk is None:
            #     self.vosk_spk = recognizer_vosk.get_vosk_spk_model(self.vosk_model)
            for i in range(len(self.vosk)):
                logger.info( f"load vosk model {i}" )
                vosk: KaldiRecognizer = KaldiRecognizer(self.vosk_model, int(self.sample_rate) )
                # if self.vosk_spk is not None:
                #     vosk.SetSpkModel( self.vosk_spk )
                vosk.SetWords(False)
                vosk.SetPartialWords(False)
                self.vosk[i] = vosk

    def start(self):
        try:
            self.stop()
            self.load()
            with self._lock:
                if self._state == 2:
                    return
                self._state = 2
            logger.info("start audio to voice")
            for i in range(len(self._thread)):
                self._thread[i] = Thread( target=self._fn_process, name="AudioToVoice", daemon=True, args=(i,))
                self._thread[i].start()
            self.audio_to_segment.start()
        except:
            logger.exception("audio to voice")

    def audio_callback(self,audio,*args):
        self.audio_to_segment.audio_callback(audio,*args)

    def _fn_callback(self, stt_data:SttData):
        with self._lock:
            stt_data.seq = self.input_count
            self.input_count+=1
            self._queue.put( stt_data )

    def _fn_process(self, no ):
        ignore_list = [ None, '', 'ん' ]
        try:
            logger.info(f"seg to voice {no} start")
            vosk:KaldiRecognizer = self.vosk[no]
            while True:
                try:
                    stt_data:SttData = self._queue.get( timeout=1.0 )
                except Empty:
                    continue
                #
                if SttData.Term==stt_data.typ:
                    self._PrioritizedCallback(stt_data,False)
                elif SttData.Segment==stt_data.typ or SttData.PreSegment == stt_data.typ:
                    w = stt_data.end - stt_data.start
                    if w<1:
                        logger.debug(f"seg_to_voice {no} ignore len:{w}")
                        self._PrioritizedCallback(stt_data,True)
                        continue
                    if vosk is not None:
                        audo_sec = w/stt_data.sample_rate
                        vosk_sec = time.time()
                        audio_i16:np.ndarray = stt_data.audio[:self.vosk_max_len] * 32767.0
                        vosk.AcceptWaveform( audio_i16.astype( np.int16 ).tobytes() )
                        vosk_res = json.loads( vosk.FinalResult())
                        vosk.Reset()
                        vosk_sec = time.time() - vosk_sec
                        txt = vosk_res.get('text')
                        if txt in ignore_list:
                            if txt is None or txt!='' or stt_data.typ==SttData.PreSegment:
                                logger.debug( f"seg_to_voice {no} vosk ignore {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                            self._PrioritizedCallback(stt_data,True)
                            continue
                        logger.debug( f"seg_to_voice {no} vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")

                    if SttData.Segment == stt_data.typ:
                        stt_data.typ = SttData.Voice
                    elif SttData.PreSegment == stt_data.typ:
                        stt_data.typ = SttData.PreVoice
                    self._PrioritizedCallback(stt_data,False)
        except:
            logger.exception("audio to voice")
        finally:
            logger.info("exit audio to voice")

    def _PrioritizedCallback(self,stt_data:SttData,ignore:bool):
        with self._lock:
            if self.output_count==stt_data.seq:
                # 順番が一致していればそのままコール
                if not ignore:
                    self.callback(stt_data)
                self.output_count=stt_data.seq+1
            else:
                # 順番が来てなければキューに入れる
                heappush( self.output_queue, (stt_data.seq,stt_data) )
            # キューを処理する
            while len(self.output_queue)>0 and self.output_queue[0][0]==self.output_count:
                _, stt_data = heappop(self.output_queue)
                if not ignore:
                    self.callback(stt_data)
                self.output_count=stt_data.seq+1

    def stop(self):
        try:
            self.audio_to_segment.stop()
            logger.info("stop audio to voice")
        except:
            logger.exception("audio to voice")
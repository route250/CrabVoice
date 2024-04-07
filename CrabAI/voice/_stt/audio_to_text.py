
from threading import Thread, Condition
from queue import Queue, Empty
import time
from logging import getLogger

try:
    from faster_whisper import WhisperModel
except:
    pass

from .stt_data import SttData
from .wave_to_audio import wave_to_audio
from .mic_to_audio import mic_to_audio
from .audio_to_voice import AudioToVoice
from .recognizer_google import RecognizerGoogle

logger = getLogger("AudioToText")

class AudioToText:

    def __init__(self, *, model:str=None, callback, sample_rate:int=16000, save_path=None):
        self._run=False
        self.model=model
        self.sample_rate = sample_rate if isinstance(sample_rate,int) else 16000
        self._state:int = 0
        self._lock:Condition = Condition()
        self._queue:Queue = Queue()
        self._thread:Thread = None
        self.callback = callback
        self.audio_to_voice:AudioToVoice = AudioToVoice( callback=self._fn_callback, save_path=save_path )
        self.whisper_model:WhisperModel = None
        self.model_size = "large-v3"
        self.w2a:wave_to_audio = None
        self.m2a:mic_to_audio = None
        # 
        self.speech_state:int = 0

    def __getitem__(self,key):
        if self.audio_to_voice is not None:
            val = self.audio_to_voice[key]
            if val is not None:
                return val
        if self.w2a is not None:
            val = self.w2a[key]
            if val is not None:
                return val
        if self.m2a is not None:
            val = self.m2a[key]
            if val is not None:
                return val
        return None

    def to_dict(self)->dict:
        ret = self.audio_to_segment.to_dict() if self.audio_to_voice is not None else {}
        if self.w2a is not None:
            ret.update( self.w2a.to_dict() )
        if self.m2a is not None:
            ret.update( self.m2a.to_dict() )
        return ret

    def __setitem__(self,key,val):
        if self.audio_to_voice is not None:
            self.audio_to_voice[key]=val
        if self.w2a is not None:
            self.w2a[key] = val
        if self.m2a is not None:
            self.m2a[key] = val

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def load(self, *, filename=None, mic=None ):
        try:
            self.audio_to_voice.load()
            if self.model=="whisper":
                if self.whisper_model is None:
                    logger.info("load audio to text")
                    self.whisper_model = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")
            else:
                pass #google

            if filename is not None:
                self.w2a:wave_to_audio = wave_to_audio( sample_rate=16000, callback=self.audio_to_voice.audio_callback )
                self.w2a.load(filename)
            elif mic is not None:
                self.m2a = mic_to_audio( sample_rate=16000, callback=self.audio_to_voice.audio_callback )
                self.m2a.load(mic=mic)
        except:
            logger.exception("audio to text")

    def start(self):
        try:
            self.stop()
            self.load()
            with self._lock:
                if self._state == 2:
                    return
                self._state = 2
            logger.info("start audio to text")
            self._thread = Thread( target=self._fn_process, name="AudioToText", daemon=True)
            self._run=True
            self._thread.start()
            self.audio_to_voice.start()
            if self.w2a is not None:
                self.w2a.start()
            elif self.m2a is not None:
                self.m2a.start()
        except:
            logger.exception("audio to text")

    def _fn_callback(self,stt_data:SttData):
        self._queue.put( stt_data )

    def _fn_process(self):
        try:
            logger.info("start audio to text")
            while self._run:
                try:
                    stt_data:SttData = self._queue.get( timeout=1.0 )
                except Empty:
                    continue
                #
                if SttData.Term == stt_data.typ:
                    self.callback(stt_data)
                    if self.speech_state==2:
                        self.speech_state=1
                elif SttData.Voice == stt_data.typ or SttData.PreVoice == stt_data.typ:
                    if self.speech_state!=2:
                        self.speech_state=2
                        self.callback( SttData( SttData.Start, stt_data.utc, stt_data.start, stt_data.start, stt_data.sample_rate, seq=stt_data.seq) )
                    audio = stt_data.audio
                    if len(audio)>0:
                        t0 = time.time()
                        if self.model=="whisper":
                            segments, info = self.whisper_model.transcribe( audio, beam_size=1, best_of=2, temperature=0, language='ja', condition_on_previous_text='まいど！' )
                            text = ""
                            for segment in segments:
                                text = text + "//" + segment.text
                        else:
                            text, confidence = RecognizerGoogle.recognize( audio, sample_rate=16000 )
                        t1 = time.time()
                        logger.debug( f"recognize {self.model} time {t1-t0:.4f}/{len(audio)/self.sample_rate:.4f}(sec)")
                    else:
                        text = ''
                    stt_data.content = text
                    if SttData.Voice == stt_data.typ:
                        stt_data.typ = SttData.Text
                    elif SttData.PreVoice == stt_data.typ:
                        stt_data.typ = SttData.PreText
                    self.callback(stt_data)
                elif SttData.Dump == stt_data.typ:
                    self.callback(stt_data)
        except:
            logger.exception("audio to text")
        finally:
            logger.info("exit audio to text")

    def tick_time(self,time_sec):
        pass

    def set_pause(self,b):
        if self.m2a is not None:
            self.m2a.set_pause(b)
        if self.w2a is not None:
            self.w2a.set_pause(b)
        self.audio_to_voice.set_pause(b)

    def stop(self):
        self._run=False
        try:
            if self._thread is not None:
                logger.info("wait for thread")
                self._thread.join(timeout=2.0)
        except:
            pass
        try:
            logger.info("stop audio to text")
        except:
            logger.exception("audio to text")

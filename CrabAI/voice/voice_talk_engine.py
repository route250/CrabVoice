import sys,os,traceback
import time, datetime
from threading import Thread, Condition
from queue import Empty

import numpy as np

from .voice_state import VoiceState
from .stt import SttEngine, SttData, get_mic_devices
from .tts import TtsEngine
from CrabAI.vmp import Ev, ShareParam
import logging
logger = logging.getLogger(__name__)

def precheck( a:bool, b:bool ) -> bool:
    return a if isinstance(a,bool) else b

class VoiceTalkEngine(ShareParam):
    EOT:str = TtsEngine.EOT
    """
    音声会話のためのエンジン。マイクから音声認識と、音声合成を行う
    """

    def __init__(self, *, conf:ShareParam=None, speaker:int=46, record_samplerate:int=1600 ):
        super().__init__(conf)
        if not isinstance(conf,ShareParam):
            SttEngine.load_default(self)
            TtsEngine.load_default(self)
        self.speaker:int = speaker
        self._share_key = self._share_array[0]
        self._status = VoiceState.ST_STOPPED
        self._callback = None
        self.text_lock:Condition = Condition()
        self.text_buf=[]
        self.text_confidence = 1.0
        self.text_stat=0
        self.th:Thread = None
        self.stt:SttEngine = None
        self.voice_id_dict:dict[float,int] = {}
        self.stt_id:int = -1
        self.stt_seq_count:dict[float,str] = {}
        self.tts:TtsEngine = None
        self.save_path:str = None
        self._input_text:list[str] = []
        #
        self.started:bool = False
        self.stopped:bool = False
        self.in_listen:bool = False
        self.in_talk:bool = False

    def __getitem__(self,key):
        if key=="save_path":
            return self.save_path
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
        keys = ['save_path']
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        if key=="save_path":
            self.save_path = val
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

    def get_speaker_id(self):
        return self.speaker

    def get_speaker_name(self):
        return TtsEngine.id_to_name(self.speaker)

    def get_speaker_gender(self):
        return TtsEngine.id_to_gender(self.speaker)

    def _fn_callback(self, stat:int, *, listen_text=None, confidence=None, talk_id=None, seq=None, talk_text=None, talk_emotion=None, talk_model=None ):
        # if stat == VoiceState.ST_LISTEN:
        #     self.play_listen_in()
        # elif stat == VoiceState.ST_LISTEN_END:
        #     self.play_listen_out()

        if self._callback is not None:
            try:
                self._callback( stat, listen_text=listen_text, confidence=None, talk_id=talk_id, seq=seq, talk_text=talk_text, talk_emotion=talk_emotion, talk_model=talk_model )
            except:
                logger.exception('')
        else:
            if stat == VoiceState.ST_LISTEN:
                logger.debug( f"[VoiceTalkEngine] listen in {listen_text} {confidence}" )
            elif stat == VoiceState.ST_LISTEN_END:
                logger.debug( f"[VoiceTalkEngine] listen out {listen_text} {confidence} __EOT__" )
            elif stat == VoiceState.ST_TALK_ENTRY:
                pass
                # logger.debug( f"[VoiceTalkEngine] talk {talk_text}" )
            elif stat == VoiceState.ST_TALK_PLAY_START:
                logger.debug( f"[VoiceTalkEngine] talk {talk_text}" )
            elif stat == VoiceState.ST_TALK_PLAY_END:
                pass
                # logger.debug( f"[VoiceTalkEngine] talk {talk_text}" )
            elif stat == VoiceState.ST_TALK_EXIT:
                logger.debug( f"[VoiceTalkEngine] talk END" )

    def load(self, *, stt=True, tts=True):
        if stt:
            mic_list = get_mic_devices(samplerate=16000, dtype=np.float32)
            if len(mic_list)>0:
                src = mic_list[0]['index']
                self.stt = SttEngine( conf=self, source=src, sample_rate=16000 )
                self.stt.load()
        if tts:
            self.tts = TtsEngine( speaker=self.speaker, talk_callback=self._tts_callback)

    def _th_loop(self):
        try:
            self.stt.start()
            while self._status != VoiceState.ST_STOPPED:
                try:
                    stt_data:SttData = self.stt.get_data( timeout=0.1 )
                    q2=True
                    # print( f"[OUT] {stt_data}")
                    if stt_data.typ == SttData.Dump:
                        self._save_audio(stt_data)
                    elif stt_data.typ == SttData.Text or stt_data.typ == SttData.Term:
                        self._fn_stt_callback( stt_data )
                    elif stt_data.typ == Ev.MuteOn or stt_data.typ == Ev.MuteOff:
                        self._fn_stt_callback( stt_data ) # STTが開始した時だけmute情報が来るはず
                except Empty:
                    q2=False

                if self.stt.is_alive():
                    continue
                else:
                    break

        except:
            logger.exception('')
        finally:
            self.stt.stop()
            self.stt.join()

    def start(self, *, stt=True):
        self._status = VoiceState.ST_LISTEN
        if stt and self.stt is not None:
            self.th:Thread = Thread( target=self._th_loop, daemon=True )
            self.th.start()

    def stop(self):
        self.stopped = True
        self._status = VoiceState.ST_STOPPED
        if self.th is not None:
            self.th.join
    
    def tick_time(self, time_sec:float):
        if self.tts is not None:
            self.tts.tick_time(time_sec)
        if self.stt is not None:
            self.stt.tick_time(time_sec)

    def add_input_text(self,message):
        with self.text_lock:
            self._input_text.append( message )

    def get_recognized_text(self):
        """音声認識による入力"""
        # logger.info("INPUT START----------------------------")
        mute,_ = self.set_mute( in_listen=True )
        while not self.stopped:
            exit_time = time.time()+2.0
            with self.text_lock:
                while not self.stopped and time.time()<exit_time:
                    if mute:
                        mute, before = self.set_mute( in_listen=True )
                        if not mute and self.tts:
                            self.tts.play_listen_in()
                    elif self.text_stat==3:
                        self.text_stat = 0
                        if len(self.text_buf)>0:
                            text = ' '.join(self.text_buf)
                            self.text_buf = []
                            confs = self.text_confidence
                            self.set_mute( in_listen=False )
                            return text, confs
                    else:
                        if self._input_text and len(self.stt_seq_count)==0:
                            start_sec = time.time()
                            stt_id = self.stt_id = self.voice_id_dict[start_sec] = len(self.voice_id_dict)
                            seq = 0
                            texts = ' '.join(self._input_text)
                            self._input_text = []
                            self.set_mute( in_listen=False )
                            for s in ( VoiceState.ST_LISTEN, VoiceState.ST_LISTEN_END):
                                self._fn_callback( s, talk_id=stt_id, seq=seq, listen_text=texts, confidence=1.0 )
                            return texts, 1.0
                        self.text_lock.wait(0.5)
            self.tick_time( time.time() )

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
        stt_id:int = self.stt_id
        seq:int = -1
        if Ev.MuteOn==typ or Ev.MuteOff==typ:
            # STTが開始した時だけmute情報が来るはず
            logger.info( f"[STT] MODULE STARTED")
            self.set_mute( started=True )
            s = VoiceState.ST_STARTED_STT
        elif SttData.Start==typ:
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} START")
            s = VoiceState.ST_LISTEN
            with self.text_lock:
                if texts is not None and len(texts)>0:
                    self.text_buf.append(texts)
                self.text_stat = 1
            copy_confidence = 1.0
        elif SttData.Text==typ:
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} {texts} {confidence}")
            s = VoiceState.ST_LISTEN
            with self.text_lock:
                if len(self.stt_seq_count)==0:
                    stt_id = self.stt_id = self.voice_id_dict[start_sec] = len(self.voice_id_dict)
                seq = self.stt_seq_count.get(start_sec)
                if seq is None:
                    seq = self.stt_seq_count[start_sec] = len(self.stt_seq_count)
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
                self.stt_seq_count = {}
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} {texts} {confidence} EOT")
            s = VoiceState.ST_LISTEN_END
            copy_confidence = self.text_confidence = confidence
        elif SttData.Dump==typ:
            return
        elif SttData.NetErr==typ:
            self.play_error1()
            return
        else:
            logger.info( f"[STT] {start_sec:.3f} - {end_sec:.3f} {stat} {texts} {confidence} EOT")
            return
        self._fn_callback( s, talk_id=stt_id, seq=seq, listen_text=texts, confidence=copy_confidence )

    def _tts_callback(self, stat:VoiceState, talk_id:int, seq:int, text:str, emotion:int, model:str):
        # voice_id
        voice_id = self.voice_id_dict.get(talk_id)
        if voice_id is None:
            voice_id = self.voice_id_dict[talk_id] = len(self.voice_id_dict)
        if isinstance(stat,VoiceState):
            if stat==VoiceState.ST_TALK_ENTRY:
                logger.debug( f"[TTS] tts_callback {stat} {text}")
                self._status = VoiceState.ST_TALK_ENTRY
                self.set_mute( in_talk=True )
                self._fn_callback( VoiceState.ST_TALK_ENTRY, talk_id=voice_id, seq=seq, talk_text=text, talk_emotion=emotion, talk_model=model )
            elif stat==VoiceState.ST_TALK_CONVERT_START or stat==VoiceState.ST_TALK_CONVERT_END:
                logger.debug( f"[TTS] tts_callback {stat} {text}")
                self._status = stat
                self._fn_callback( stat, talk_id=voice_id, seq=seq, talk_text=text, talk_emotion=emotion, talk_model=model )
            elif stat==VoiceState.ST_TALK_PLAY_START or stat==VoiceState.ST_TALK_PLAY_END:
                logger.debug( f"[TTS] tts_callback {stat} {text}")
                self._status = stat
                self._fn_callback( stat, talk_id=voice_id, seq=seq, talk_text=text, talk_emotion=emotion, talk_model=model )
            elif stat==VoiceState.ST_TALK_EXIT:
                logger.debug( f"[TTS] tts_callback {stat} {text}")
                self._status = VoiceState.ST_LISTEN
                self._fn_callback( VoiceState.ST_TALK_EXIT, talk_id=voice_id, talk_text=None )
                self.set_mute( in_talk=False ) #pp
            else:
                logger.error( f"[TTS] tts_callback {stat}")

    def set_mute(self, *, started=None, in_talk=None, in_listen=None ):
        with self.text_lock:
            self.started = precheck( started, self.started )
            self.in_talk = precheck( in_talk, self.in_talk )
            self.in_listen = precheck( in_listen, self.in_listen )
            mute = not self.started or self.in_talk or not self.in_listen
            if self.stt:
                before = self.stt.get_mute()
                # if mute != before and not mute:
                #     self.play_mute_out(wait=True)
                x3,x4 = self.stt.set_mute(mute)
                if x3!=x4 and x3:
                    self.play_mute_in()
                return x3,x4
        return False,False

        # if after != before and not after:
        #     stack = traceback.extract_stack()
        #     filtered_stack = [frame for frame in stack if 'maeda/LLM/CrabVoice/' in frame.filename]
        #     stack_trace = ''.join(traceback.format_list(filtered_stack))
        #     logger.info(f"[STT] set_mute in_talk={in_talk} in_listen={in_listen}\n%s", stack_trace)

    def play_mute_in(self):
        if self.tts is not None:
            self.tts.play_mute_in()

    def play_mute_out(self,wait=False):
        if self.tts is not None:
            self.tts.play_mute_out(wait=wait)

    def play_listen_in(self):
        if self.tts is not None:
            self.tts.play_listen_in()

    def play_listen_out(self):
        if self.tts is not None:
            self.tts.play_listen_out()

    def play_error1(self):
        if self.tts is not None:
            self.tts.play_error1()

    def play_error2(self):
        if self.tts is not None:
            self.tts.play_error2()

    def add_talk(self, text ):
        # if self.stt is not None:
        #     self.stt.set_pause( True )
        if self.tts is not None:
            self.tts.add_talk( text )
        else:
            print( f"[TTS]add_talk {text}" )

    def sep_talk(self):
        self.tts.cancel()

    def _save_audio(self,stt_data:SttData):
        try:
            if self.save_path is not None and os.path.isdir(self.save_path):
                max_vad = max(stt_data['vad'])
                if max_vad>0.5:
                    logger.info( f"[DUMP] {max_vad} {stt_data}")
                    dt = datetime.datetime.now()
                    dir = os.path.join( self.save_path, dt.strftime("%Y%m%d") )
                    os.makedirs(dir,exist_ok=True)
                    filename = dt.strftime("audio_%Y%m%d_%H%M%S")
                    file_path = os.path.join( dir, filename )
                    stt_data.save(file_path)
                # else:
                #     print( f"[dump] {stt_data}")                    
        except:
            logger.exception("can not save data")
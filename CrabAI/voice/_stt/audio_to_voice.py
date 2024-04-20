from logging import getLogger
logger = getLogger("AudioToVoide")

import sys,os,datetime
from threading import Thread, Condition
from queue import Queue, PriorityQueue, Empty
import time
import json
from heapq import heapify, heappop, heappush

import numpy as np
_X_VOSK_:bool=False
try:
    import vosk
    from vosk import KaldiRecognizer
    vosk.SetLogLevel(-1)
    from . import recognizer_vosk
    _X_VOSK:bool=True
except:
    _X_VOSK:bool=False
    pass
# import librosa
# import scipy

from .stt_data import SttData

#from .audio_to_segment_webrtcvad import AudioToSegmentWebrtcVAD as AudioToSegment
from .audio_to_segment_silero_vad import AudioToSegmentSileroVAD as AudioToSegment
from ..voice_utils import voice_per_audio_rate

class AudioToVoice:
    def __init__(self, *, callback, sample_rate:int=16000, save_path=None):
        self.sample_rate = sample_rate if isinstance(sample_rate,int) else 16000
        self._state:int = 0
        self._lock:Condition = Condition()
        self._queue:Queue = Queue()
        self.vad_vosk = _X_VOSK
        self.num_threads:int = 2
        self._thread:list[Thread] = [None] * self.num_threads
        self.callback = callback
        self._mute:bool = False # ミュートする
        self._var3:float = 0.45
        self.audio_to_segment:AudioToSegment = AudioToSegment( callback=self._fn_callback )
        self.vosk_gr: list[KaldiRecognizer] = [None] * len(self._thread)
        self.vosk2: list[KaldiRecognizer] = [None] * len(self._thread)
        self.vosk_model: vosk.Model = None
        self.vosk_spk: vosk.SpkModel = None
        self.vosk_max_len:int = int(self.sample_rate*1.7)
        # 
        self.input_count:int = 0
        self.output_count:int = 0
        self.output_queue:list = []
        heapify(self.output_queue)
        # 人の声のフィルタリング（バンドパスフィルタ）
        # fs_nyq = self.sample_rate*0.5
        # low = 200 / fs_nyq
        # high = 1000 /fs_nyq
        # self.pass_ba = scipy.signal.butter( 2, [low, high], 'bandpass', output='ba')
        # self.cut_ba = scipy.signal.butter( 2, [low, high], 'bandstop', output='ba')
        # データ保存用
        self.save_path:str = save_path

    def __getitem__(self,key):
        if 'mute'==key:
            return self._mute
        elif 'var3'==key:
            return self._var3
        elif 'vad.vosk'==key:
            return self.vad_vosk
        elif 'save_path'==key:
            return self._var3
        return self.audio_to_segment[key]

    def to_dict(self)->dict:
        keys = ['mute','var3','save_path']
        ret = self.audio_to_segment.to_dict()
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        if 'mute'==key:
            if isinstance(val,(bool)):
                self.set_pause(val)
        elif 'var3'==key:
            if isinstance(val,(int,float)) and 0<=val<=1:
                self._var3 = float(val)
        elif 'vad.vosk'==key:
            if isinstance(val,(bool)):
                self.vad_vosk=val
        elif 'save_path'==key:
            if isinstance(val,str):
                self.save_path = val
        else:
            self.audio_to_segment[key] = val

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def load(self):
        self.audio_to_segment.load()
        if self.num_threads>0 and self.vosk_gr[0] is None:
            if self.vosk_model is None:
                logger.info( f"load vosk lang model" )
                self.vosk_model = recognizer_vosk.get_vosk_model(lang="ja")
            # if self.vosk_spk is None:
            #     self.vosk_spk = recognizer_vosk.get_vosk_spk_model(self.vosk_model)
            katakana_grammer:str = recognizer_vosk.get_katakana_grammar()
            for i in range(len(self.vosk_gr)):
                logger.info( f"load vosk model {i}" )
                if katakana_grammer is not None:
                    vosk: KaldiRecognizer = KaldiRecognizer(self.vosk_model, int(self.sample_rate), katakana_grammer )
                    # if self.vosk_spk is not None:
                    #     vosk.SetSpkModel( self.vosk_spk )
                    self.vosk_gr[i] = vosk
                else:
                    self.vosk_gr[i] = None
                vosk: KaldiRecognizer = KaldiRecognizer(self.vosk_model, int(self.sample_rate) )
                # if self.vosk_spk is not None:
                #     vosk.SetSpkModel( self.vosk_spk )
                self.vosk2[i] = vosk

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
            if self._mute:
                return
            stt_data.seq = self.input_count
            self.input_count+=1
            self._queue.put( stt_data )

    def _fn_process(self, no ):
        ignore_list = [ None, '', 'ん' ]
        try:
            logger.info(f"seg to voice {no} start")
            grammer:KaldiRecognizer = self.vosk_gr[no]
            vosk2:KaldiRecognizer = self.vosk2[no]
            while True:
                try:
                    stt_data:SttData = self._queue.get( timeout=1.0 )
                except Empty:
                    continue
                #
                if SttData.Segment==stt_data.typ or SttData.PreSegment == stt_data.typ:
                    Accept:bool = None
                    stt_length = stt_data.end - stt_data.start
                    stt_audio = stt_data.audio
                    audo_sec = stt_length/stt_data.sample_rate

                    if stt_length<1 or stt_audio is None:
                        Accept = False
                        logger.debug(f"seg_to_voice {no} ignore len:{stt_length}")
                        print(f"[seg_to_voice] {no} ignore len:{stt_length}")

                    if Accept is None:
                        # 音量調整
                        peek = np.max(stt_audio)
                        if peek<0.8:
                            stt_audio = stt_audio * (0.8/peek)

                    if Accept is None and grammer:
                        vosk_sec = time.time()
                        audio_i16:np.ndarray = stt_audio * 32767.0
                        audio_bytes:bytes = audio_i16.astype( np.int16 ).tobytes()
                        grammer.AcceptWaveform( audio_bytes )
                        vosk_res = json.loads( grammer.FinalResult())
                        grammer.Reset()
                        vosk_sec = time.time() - vosk_sec
                        txt = vosk_res.get('text')
                        if txt in ignore_list:
                            logger.debug( f"seg_to_voice {no} reject grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                            print( f"[seg_to_voice] {no} reject grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                        else:
                            Accept = True
                            logger.debug( f"seg_to_voice {no} accept grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                            print( f"[seg_to_voice] {no} accept grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")

                    if Accept is None:
                        # 判定する範囲
                        hist_vad:np.ndarray = stt_data['vad']
                        fz:int = stt_length // len(hist_vad)
                        center:int = hist_vad.argmax() * fz

                    if Accept is None:
                        ast = max( 0, center - int(self.vosk_max_len*0.3) )
                        aed = min( len(stt_audio), ast + self.vosk_max_len )
                        # FFT判定
                        var = voice_per_audio_rate( stt_audio[ast:aed], sampling_rate=16000 )
                        if var<self._var3:
                            Accept = False
                            logger.debug(f"seg_to_voice {no} reject voice/audio {var}")
                            print(f"[seg_to_voice] {no} reject voice/audio {var}")
                        else:
                            print( f"[seg_to_voice] {no} accept voice/audio {var}" )
                    #
                    if Accept is None and self.vad_vosk and vosk2 is not None:
                        # 判定する範囲
                        ast = max( 0, center - int(self.vosk_max_len*0.3) )
                        aed = min( len(stt_audio), ast + self.vosk_max_len )
                        vosk_sec = time.time()
                        audio_i16:np.ndarray = stt_audio[ast:aed] * 32767.0
                        audio_bytes:bytes = audio_i16.astype( np.int16 ).tobytes()
                        vosk2.AcceptWaveform( audio_bytes )
                        vosk_res = json.loads( vosk2.FinalResult())
                        vosk2.Reset()
                        vosk_sec = time.time() - vosk_sec
                        txt = vosk_res.get('text')
                        if txt in ignore_list:
                            logger.debug( f"seg_to_voice {no} reject vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                            print( f"[seg_to_voice] {no} reject vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                        else:
                            Accept = True
                            logger.debug( f"seg_to_voice {no} accept vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                            print( f"[seg_to_voice] {no} accept vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")

                    if Accept is not None and Accept:
                        if SttData.Segment == stt_data.typ:
                            stt_data.typ = SttData.Voice
                        elif SttData.PreSegment == stt_data.typ:
                            stt_data.typ = SttData.PreVoice
                        self._PrioritizedCallback(stt_data,False)
                    else:
                        self._PrioritizedCallback(stt_data,True)

                elif SttData.Term==stt_data.typ:
                    self._PrioritizedCallback(stt_data,False)
                elif SttData.Dump==stt_data.typ:
                    self._PrioritizedCallback(stt_data,False)
                    self._save_audio(stt_data)
                else:
                    self._PrioritizedCallback(stt_data,True)
        except:
            logger.exception("audio to voice")
        finally:
            logger.info("exit audio to voice")

    def _PrioritizedCallback(self,stt_data:SttData,ignore:bool):
        with self._lock:
            if self._mute:
                return
            if self.output_count==stt_data.seq:
                # 順番が一致していればそのままコール
                if not ignore:
                    self.callback(stt_data)
                self.output_count=stt_data.seq+1
            else:
                # 順番が来てなければキューに入れる
                if not ignore:
                    heappush( self.output_queue, (stt_data.seq,stt_data) )
                else:
                    heappush( self.output_queue, (stt_data.seq,None) )
            # キューを処理する
            while len(self.output_queue)>0 and self.output_queue[0][0]==self.output_count:
                seq, stt_data = heappop(self.output_queue)
                if stt_data is not None:
                    self.callback(stt_data)
                self.output_count=seq+1

    def set_pause(self,b):
        try:
            with self._lock:
                before = self._mute
                self._mute = bool(b)
                if not before and b:
                    self._queue = Queue()
                    self.input_count:int = 0
                    self.output_count:int = 0
                    self.output_queue:list = []
                    heapify(self.output_queue)
            self.audio_to_segment.set_pause(b)
        except:
            pass

    def stop(self):
        try:
            self.audio_to_segment.stop()
            logger.info("stop audio to voice")
        except:
            logger.exception("audio to voice")

    def _save_audio(self,stt_data:SttData):
        try:
            if self.save_path is not None and os.path.isdir(self.save_path):
                max_vad = max(stt_data['vad'])
                if max_vad>0.2:
                    dt = datetime.datetime.now()
                    dir = os.path.join( self.save_path, dt.strftime("%Y%m%d") )
                    os.makedirs(dir,exist_ok=True)
                    filename = dt.strftime("audio_%Y%m%d_%H%M%S")
                    file_path = os.path.join( dir, filename )
                    stt_data.save(file_path)
        except:
            logger.exception("can not save data")
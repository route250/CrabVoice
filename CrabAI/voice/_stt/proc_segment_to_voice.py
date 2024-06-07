import sys,os
import platform
from logging import getLogger
import numpy as np
from multiprocessing import Queue

from logging import getLogger
logger = getLogger(__name__)

import time
import json

import numpy as np
from scipy import signal
_X_VOSK_:bool=False
try:
    import vosk
    vosk.SetLogLevel(-1)
    from vosk import KaldiRecognizer
    from . import recognizer_vosk
    _X_VOSK:bool=True
except:
    _X_VOSK:bool=False

from CrabAI.vmp import Ev, VFunction, VProcess
from .stt_data import SttData
from ..voice_utils import voice_per_audio_rate

class SegmentToVoice(VFunction):
    DEFAULT_BUTTER = [ 100, 10, 10, 90 ] # fpass, fstop, gpass, gstop
    def __init__(self, proc_no:int, num_proc:int, share, data_in:Queue, data_out:Queue, *, sample_rate:int=None ):
        super().__init__(proc_no,num_proc,share,data_in,data_out)
        self.state:int = 0
        self.sample_rate:int = sample_rate if isinstance(sample_rate,int) else 16000

        self._state:int = 0
        self.vad_vosk = _X_VOSK
        self._var3:float = self.conf.set_voice_var(0.45,notify=False)
        self.vosk_gr: KaldiRecognizer = None
        self.vosk_recog: KaldiRecognizer = None
        self.vosk_model: vosk.Model = None
        self.vosk_spk: vosk.SpkModel = None
        self.vosk_grammar:str = None
        self.vosk_max_len:int = int( self.conf.set_voice_max_sec(1.7,notify=False) * self.sample_rate )
        # 
        # 人の声のフィルタリング（バンドパスフィルタ）
        # fs_nyq = self.sample_rate*0.5
        # low = 200 / fs_nyq
        # high = 1000 /fs_nyq
        # self.pass_ba = scipy.signal.butter( 2, [low, high], 'bandpass', output='ba')
        # self.cut_ba = scipy.signal.butter( 2, [low, high], 'bandstop', output='ba')
        self._butter = self.conf.set_butter2( SegmentToVoice.DEFAULT_BUTTER, notify=False )
        self.sos = None
        self._update_butter()
        self.ignore_list = [ None, '', 'ん' ]

    def _update_butter(self):
        fpass, fstop, gpass, gstop = self._butter
        fpass2 = np.array([fpass,7000])
        fstop2 = np.array([fstop,8000])
        fn = self.sample_rate // 2   #ナイキスト周波数
        wp = fpass2 / fn  #ナイキスト周波数で通過域端周波数を正規化
        ws = fstop2 / fn  #ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
        # self.b, self.a = signal.butter(N, Wn, "high")   #フィルタ伝達関数の分子と分母を計算
        self.sos = signal.butter(N, Wn, "band", output='sos')   #フィルタ伝達関数の分子と分母を計算

    def audio_filter(self,x):
        if not isinstance(x,np.ndarray) or len(x)==0:
            return x
        #y = signal.filtfilt(self.b, self.a, x) #信号に対してフィルタをかける
        y:np.ndarray = signal.sosfiltfilt( self.sos, x ) #信号に対してフィルタをかける
        return y.astype(np.float32)

    def load(self):
        if self.vad_vosk:
            vosk.SetLogLevel(-1)
            logger.info( f"load vosk lang model" )
            self.vosk_model = recognizer_vosk.get_vosk_model(lang="ja")
            logger.info( f"load vosk spk model" )
            self.vosk_spk = recognizer_vosk.get_vosk_spk_model(self.vosk_model)
            logger.info( f"load vosk grammar" )
            self.vosk_grammar:str = recognizer_vosk.get_katakana_grammar()

            if self.vosk_grammar is not None:
                logger.info( f"create grammar model" )
                recog: KaldiRecognizer = KaldiRecognizer(self.vosk_model, int(self.sample_rate), self.vosk_grammar )
                if self.vosk_spk is not None:
                    recog.SetSpkModel( self.vosk_spk )
                self.vosk_gr = recog
            else:
                self.vosk_gr = None

            logger.info( f"create vosk model" )
            recog: KaldiRecognizer = KaldiRecognizer(self.vosk_model, int(self.sample_rate) )
            if self.vosk_spk is not None:
                recog.SetSpkModel( self.vosk_spk )
            self.vosk_recog = recog

    def reload_share_param(self):
        butter = self.conf.get_butter2()
        if isinstance(butter,list) and len(butter)==len(self._butter) and butter != self._butter:
            self._butter = butter
            self._update_butter()
        self._var3 = self.conf.get_voice_var()
        self.vosk_max_len = int( self.conf.get_voice_max_sec() * self.sample_rate )

    def proc(self, ev ):
        if isinstance(ev,SttData):
            if SttData.Segment==ev.typ or SttData.PreSegment == ev.typ:
                self.proc_segment(ev)
            else:
                self.proc_output_event(ev)
        else:
            if Ev.EndOfData==ev.typ:
                pass # このメソッドの外側でoutput_evされる
            else:
                self.proc_output_event(ev)

    def proc_segment(self, stt_data:SttData):
        no = self.proc_no
        ignore_list = self.ignore_list
        grammer:KaldiRecognizer = self.vosk_gr
        vosk2:KaldiRecognizer = self.vosk_recog
        try:
            if stt_data.sample_rate != self.sample_rate:
                return # finallyで処理する

            Accept:bool = None
            stt_length = stt_data.end - stt_data.start
            stt_audio = self.audio_filter(stt_data.raw)
            audo_sec = stt_length/stt_data.sample_rate

            if stt_length<1 or stt_audio is None:
                Accept = False
                logger.debug(f"seg_to_voice {no} ignore len:{stt_length}")
                print(f"[seg_to_voice] {no} ignore len:{stt_length}")

            if Accept is None:
                # 音量調整
                peek = np.max(stt_audio)
                if 0<peek and peek<0.4:
                    stt_audio = stt_audio * (0.4/peek)

            if Accept is None and grammer:
                try:
                    #print(f"[seg_to_voice] {no} #1-4-1")
                    #vosk.SetLogLevel(1)
                    # 判定する範囲
                    vosk_sec = time.time()
                    # vadが一番高いところを中心に判定する
                    hist_vad:np.ndarray = stt_data['vad']
                    fz:int = stt_length // len(hist_vad)
                    center:int = hist_vad.argmax() * fz
                    ast = max( 0, center - int(self.vosk_max_len*0.6) )
                    aed = min( len(stt_audio), ast + self.vosk_max_len*2 )
                    audio_i16:np.ndarray = stt_audio * 32766.0
                    audio_bytes:bytes = audio_i16.astype( np.int16 ).tobytes()
                    #print(f"[set_to_voice] {no} #1-4-2")
                    grammer.AcceptWaveform( audio_bytes )
                    vosk_res = json.loads( grammer.FinalResult())
                    grammer.Reset()
                    vosk_sec = time.time() - vosk_sec
                    spk = vosk_res.get('spk')
                    if isinstance(spk,list):
                        stt_data.spk = np.array(spk)
                        vosk_res['spk'] = True
                    txt = vosk_res.get('text','')
                    if txt in ignore_list:
                        logger.debug( f"seg_to_voice {no} reject grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                        print( f"[seg_to_voice] {no} reject grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                    else:
                        Accept = True
                        stt_data.content = txt
                        logger.debug( f"seg_to_voice {no} accept grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                        print( f"[seg_to_voice] {no} accept grammer {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                except Exception as err:
                    logger.exception(f"#4 {str(err)}")
                finally:
                    vosk.SetLogLevel(-1)
                    #print(f"[set_to_voice] {no} #1-4-99")
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
                audio_i16:np.ndarray = stt_audio[ast:aed] * 32766.0
                audio_bytes:bytes = audio_i16.astype( np.int16 ).tobytes()
                vosk2.AcceptWaveform( audio_bytes )
                vosk_res = json.loads( vosk2.FinalResult())
                vosk2.Reset()
                vosk_sec = time.time() - vosk_sec
                spk = vosk_res.get('spk')
                if isinstance(spk,list):
                    stt_data.spk = np.array(spk)
                    vosk_res['spk'] = True
                txt = vosk_res.get('text')
                if txt in ignore_list:
                    logger.debug( f"seg_to_voice {no} reject vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                    print( f"[seg_to_voice] {no} reject vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                else:
                    Accept = True
                    stt_data.content = txt
                    logger.debug( f"seg_to_voice {no} accept vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")
                    print( f"[seg_to_voice] {no} accept vosk {vosk_sec:.4f}(sec)/{audo_sec:.4f} {vosk_res}")

            if Accept is not None and Accept:
                if SttData.Segment == stt_data.typ:
                    stt_data.typ = SttData.Voice
                elif SttData.PreSegment == stt_data.typ:
                    stt_data.typ = SttData.PreVoice
                self.proc_output_event(stt_data)
                stt_data = None
        except:
            logger.exception("audio to voice")
        finally:
            if stt_data is not None:
                stt_data.typ = Ev.Nop
                self.proc_output_event(stt_data)



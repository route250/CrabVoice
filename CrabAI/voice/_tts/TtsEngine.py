import sys,os,traceback,json,re
from threading import Thread, Condition, ThreadError
from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing import Queue
from queue import Empty
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
import httpx

import openai
from openai import OpenAI

from gtts import gTTS
from io import BytesIO
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import wave
import librosa

from ...net.net_utils import find_first_responsive_host
from ..voice_utils import mml_to_audio, audio_to_wave_bytes, create_tone
from ..translate import convert_to_katakana, convert_kuten
from CrabAI.vmp import ShareParam

from logging import getLogger
logger = getLogger(__name__)

class TtsEngine:
    EOT:str = "<|EOT|>"
    VoiceList = [
        ( "VOICEVOX:四国めたん [あまあま]", 0, 'ja_JP' ),
        ( "VOICEVOX:四国めたん [ノーマル]", 2, 'ja_JP' ),
        ( "VOICEVOX:四国めたん [セクシー]", 4, 'ja_JP' ),
        ( "VOICEVOX:四国めたん [ツンツン]", 6, 'ja_JP' ),
        ( "VOICEVOX:ずんだもん [あまあま]", 1, 'ja_JP' ),
        ( "VOICEVOX:ずんだもん [ノーマル]", 3, 'ja_JP' ),
        ( "VOICEVOX:ずんだもん [セクシー]", 5, 'ja_JP' ),
        ( "VOICEVOX:ずんだもん [ツンツン]", 7, 'ja_JP' ),
        ( "VOICEVOX:春日部つむぎ [ノーマル]",8, 'ja_JP' ),
        ( "VOICEVOX:波音リツ [ノーマル]", 9, 'ja_JP' ),
        ( "VOICEVOX:雨晴はう [ノーマル]", 10, 'ja_JP' ),
        ( "VOICEVOX:玄野武宏 [ノーマル]", 11, 'ja_JP' ),
        ( "VOICEVOX:白上虎太郎 [ふつう]", 11, 'ja_JP' ),
        ( "VOICEVOX:白上虎太郎 [わーい]", 32, 'ja_JP' ),
        ( "VOICEVOX:白上虎太郎 [びくびく]", 33, 'ja_JP' ),
        ( "VOICEVOX:白上虎太郎 [おこ]", 34, 'ja_JP' ),
        ( "VOICEVOX:白上虎太郎 [びえーん]", 36, 'ja_JP' ),
        ( "VOICEVOX:冥鳴ひまり [ノーマル]", 14, 'ja_JP' ),
        ( "VOICEVOX:もち子(cv 明日葉よもぎ)[ノーマル]", 20, 'ja_JP' ),
        ( "VOICEVOX:小夜/SAYO [ノーマル]", 46, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [ノーマル]", 13, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [熱血]", 81, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [不機嫌]", 82, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [喜び]", 83, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [しっとり]", 84, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [かなしみ]", 85, 'ja_JP' ),
        ( "VOICEVOX:青山龍星 [囁き]", 86, 'ja_JP' ),
        ( "VOICEVOX:剣崎雌雄 [ノーマル]", 21, 'ja_JP' ),
        ( "VOICEVOX:小夜/SAYO [ノーマル]", 46, 'ja_JP' ),
        ( "VOICEVOX:雀松朱司 [ノーマル]", 52, 'ja_JP' ),
        ( "OpenAI:alloy", 1001, 'ja_JP' ),
        ( "OpenAI:echo", 1002, 'ja_JP' ),
        ( "OpenAI:fable", 1003, 'ja_JP' ),
        ( "OpenAI:onyx", 1004, 'ja_JP' ), # 男性っぽい
        ( "OpenAI:nova", 1005, 'ja_JP' ), # 女性っぽい
        ( "OpenAI:shimmer", 1006, 'ja_JP' ), # 女性ぽい
        ( "gTTS:[ja_JP]", 2000, 'ja_JP' ),
        ( "gTTS:[en_US]", 2001, 'en_US' ),
        ( "gTTS:[en_GB]", 2002, 'en_GB' ),
        ( "gTTS:[fr_FR]", 2003, 'fr_FR' ),
    ]

    @staticmethod
    def id_to_model( idx:int ) -> str:
        return next((voice for voice in TtsEngine.VoiceList if voice[1] == idx), None )

    @staticmethod
    def id_to_name( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        name = voice[0]
        return name if name else '???'

    @staticmethod
    def id_to_lang( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        lang = voice[2]
        return lang if lang else 'ja_JP'

    @staticmethod
    def load_default( conf:ShareParam ):
        pass

    def __init__(self, *, speaker=-1, submit_task = None, talk_callback = None, katakana_dir='tmp/katakana' ):
        # 並列処理用
        self.lock:Condition = Condition()
        self._running_future:Future = None
        self._running_future2:Future = None
        self.wave_queue:Queue = Queue()
        self.play_queue:Queue = Queue()
        self._last_talk:float = 0
        # 発声中のセリフのID
        self._talk_id: int = 0
        # 音声エンジン選択
        self.speaker = speaker
        # コールバック
        self.executor = None
        self.submit_call = submit_task # スレッドプールへの投入
        self.start_call = talk_callback # 発声開始と完了を通知する
        # pygame初期化済みフラグ
        self.pygame_init:bool = False
        # beep
        self.beep_ch:pygame.mixer.Channel  = None
        # 音声エンジン無効時間
        self._disable_gtts: float = 0.0
        self._disable_openai: float = 0.0
        self._disable_voicevox: float = 0.0
        # VOICEVOXサーバURL
        self._voicevox_url = None
        self._voicevox_port = os.getenv('VOICEVOX_PORT','50021')
        self._voicevox_list = list(set([os.getenv('VOICEVOX_HOST','127.0.0.1'),'127.0.0.1','192.168.0.104','chickennanban.ddns.net','chickennanban1.ddns.net','chickennanban2.ddns.net','chickennanban3.ddns.net']))
        self._katakana_dir = katakana_dir

        self.feed = create_tone( 32, time=0.4, volume=0.9, sample_rate=16000)
        self.feed_wave = audio_to_wave_bytes(self.feed, sample_rate=16000 )
        self.sound_listen_in = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 ce", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_listen_out = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 ec", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_error1 = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t240v15 O3aa", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_error2 = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v15 O3aaa", sampling_rate=16000 ))), sample_rate=16000 )

    def __getitem__(self,key):
        if 'speaker.id'==key:
            return self.speaker
        elif 'speaker.name'==key:
            return TtsEngine.id_to_name(self.speaker)
        elif 'disable.gtts'==key:
            return self._disable_gtts
        elif 'disable.openai'==key:
            return self._disable_openai
        elif 'disable.voicevox'==key:
            return self._disable_voicevox
        return None

    def to_dict(self)->dict:
        keys = ['speaker.id','speaker.name','disable.gtts','disable.openai','disable.voicevox']
        ret = {}
        for key in keys:
            ret[key] = self[key]
        return ret

    def __setitem__(self,key,val):
        if 'speaker.id'==key:
            if isinstance(val,(int,float)) and 0<=key<=3:
                self.vad_mode = int(key)

    def update(self,arg=None,**kwargs):
        upd = {}
        if isinstance(arg,dict):
            upd.update(arg)
        upd.update(kwargs)
        for key,val in upd.items():
            self[key]=val

    def tick_time(self, time_sec:float ):
        pass

    def _sound_init(self):
        try:
            if self.pygame_init and (time.time()-self._last_talk)>100.0:
                self.pygame_init = False
                print(f"[PyGame]reset")
                pygame.mixer.quit()
        except:
            logger.exception("can not reset pygame")
        try:
            if not self.pygame_init:
                print(f"[PyGame]init")
                pygame.mixer.pre_init(16000,-16,1,10240)
                pygame.mixer.quit()
                pygame.mixer.init()
                self.pygame_init = True
        except:
            logger.exception("can not reset pygame")
        # if not self.pygame_init:
        #     pygame.init()
        #     pygame.mixer.pre_init(16000,-16,1,10240)
        #     pygame.mixer.quit()
        #     pygame.mixer.init()
        #     self.pygame_init = True

    def submit_task(self, func ) -> Future:
        if self.submit_call is not None:
            return self.submit_call(func)
        if self.executor is None:
            self.executor:ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
        return self.executor.submit( func )

    def cancel(self):
        self._talk_id += 1

    def _get_voicevox_url( self ) ->str:
        if self._voicevox_url is None:
            self._voicevox_url = find_first_responsive_host(self._voicevox_list,self._voicevox_port)
        return self._voicevox_url

    @staticmethod
    def remove_code_blocksRE(markdown_text):
        # 正規表現を使用してコードブロックを検出し、それらを改行に置き換えます
        # ```（コードブロックの開始と終了）に囲まれた部分を検出します
        # 正規表現のパターンは、```で始まり、任意の文字（改行を含む）にマッチし、最後に```で終わるものです
        # re.DOTALLは、`.`が改行にもマッチするようにするフラグです
        pattern = r'```.*?```'
        return re.sub(pattern, '\n', markdown_text, flags=re.DOTALL)

    @staticmethod
    def split_talk_text( text):
        sz = len(text)
        st = 0
        lines = []
        while st<sz:
            block_start = text.find("```",st)
            newline_pos = text.find('\n',st)
            if block_start>=0 and ( newline_pos<0 or block_start<newline_pos ):
                if st<block_start:
                    lines.append( text[st:block_start] )
                block_end = text.find( "```", block_start+3)
                if (block_start+3)<block_end:
                    block_end += 3
                else:
                    block_end = sz
                lines.append( text[block_start:block_end])
                st = block_end
            else:
                if newline_pos<0:
                    newline_pos = sz
                if st<newline_pos:
                    lines.append( text[st:newline_pos] )
                st = newline_pos+1
        return lines

    def add_talk(self, full_text:str, emotion:int = 0 ) -> None:
        talk_id:int = self._talk_id
        for text in TtsEngine.split_talk_text(full_text):
            print(f"[TTS] put {text}")
            self.wave_queue.put( (talk_id, text, emotion ) )
        with self.lock:
            if self._running_future is None:
                self._running_future = self.submit_task(self.run_text_to_audio)
    
    def run_text_to_audio(self)->None:
        """ボイススレッド
        テキストキューからテキストを取得して音声に変換して発声キューへ送る
        """
        print(f"[TTS] thread start")
        while True:
            talk_id:int = -1
            text:str = None
            emotion:int = -1
            with self.lock:
                try:
                    talk_id, text, emotion = self.wave_queue.get_nowait()
                except Exception as ex:
                    if not isinstance( ex, Empty ):
                        logger.exception(ex)
                    talk_id=-1
                    text = None
                if text is None:
                    self._running_future = None
                    print(f"[TTS] thread end")
                    return
            try:
                print(f"[TTS] text {text}")
                if talk_id == self._talk_id:
                    # textから音声へ
                    audio_bytes, tts_model = self._text_to_audio( text, emotion )
                    self._add_audio( talk_id,text,emotion,audio_bytes,tts_model )
            except Exception as ex:
                logger.exception(ex)

    def _add_audio( self, talk_id:int, text:str, emotion:int, audio_bytes: bytes, tts_model:str=None ) -> None:
        self.play_queue.put( (talk_id,text,emotion,audio_bytes,tts_model) )
        with self.lock:
            if self._running_future2 is None:
                self._running_future2 = self.submit_task(self.run_talk)

    @staticmethod
    def __penpenpen( text, default=" " ) ->str:
        if text is None or text.startswith("```"):
            return default # VOICEVOX,OpenAI,gTTSで、エラーにならない無音文字列
        else:
            return text
        
    def _text_to_audio_by_voicevox(self, text: str, emotion:int = 0, lang='ja') -> bytes:
        if self._disable_voicevox>0 and (time.time()-self._disable_voicevox)<180.0:
            return None,None
        sv_url: str = self._get_voicevox_url()
        if sv_url is None:
            self._disable_voicevox = time.time()
            return None,None
        try:
            text = convert_to_katakana(text,cache_dir=self._katakana_dir)
            text = convert_kuten(text)
            text = TtsEngine.__penpenpen(text, ' ')
            self._disable_voicevox = 0
            timeout = (5.0,180.0)
            params = {'text': text, 'speaker': self.speaker, 'timeout': timeout }
            s:requests.Session = requests.Session()
            s.mount(f'{sv_url}/audio_query', HTTPAdapter(max_retries=1))
            res1 : requests.Response = requests.post( f'{sv_url}/audio_query', params=params)
            data = res1.content
            res1_json:dict = json.loads(data)
            ss:float = res1_json.get('speedScale',1.0)
            res1_json['speedScale'] = ss*1.2
            ps:float = res1_json.get('pitchScale',0.0)
            res1_json['pitchScale'] = ps-0.1
            data = json.dumps(res1_json,ensure_ascii=False)
            params = {'speaker': self.speaker, 'timeout': timeout }
            headers = {'content-type': 'application/json'}
            res = requests.post(
                f'{sv_url}/synthesis',
                data=data,
                params=params,
                headers=headers
            )
            model:str = TtsEngine.id_to_name(self.speaker)
            # wave形式 デフォルトは24kHz
            return res.content, model
        except requests.exceptions.ConnectTimeout as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
        except requests.exceptions.ConnectionError as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
        except Exception as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
            logger.exception('')
        self._disable_voicevox = time.time()
        return None,None

    def _text_to_audio_by_gtts(self, text: str, emotion:int = 0) -> bytes:
        if self._disable_gtts>0 and (time.time()-self._disable_gtts)<180.0:
            return None,None
        voice = TtsEngine.id_to_model( self.speaker )
        lang = voice[2] if voice else 'ja_JP'
        lang = lang[:2]
        try:
            self._disable_gtts = 0
            tts:gTTS = gTTS(text=TtsEngine.__penpenpen(text,'!!'), lang=lang,lang_check=False )
            with BytesIO() as data_mp3:
                tts.write_to_fp(data_mp3)
                try:
                    # gTTSはmp3で返ってくるので変換
                    data_mp3.seek(0)
                    y, sr = librosa.load(data_mp3, sr=None)
                    if lang=='ja':
                        sr = int( sr*1.7 )
                        # ピッチを下げる
                        n_steps = -8
                        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                    data:bytes = audio_to_wave_bytes(y, sample_rate=sr )
                except:
                    logger.exception('convert gtts mp3 to wave')
                    data_mp3.seek(0)
                    data:bytes = data_mp3.getvalue()
            del tts
            return data,f"gTTS[{lang}]"
        except AssertionError as ex:
            if "No text to send" in str(ex):
                return None,f"gTTS[{lang}]"
            logger.error( f"[gTTS] {ex}")
            logger.exception('')
        except requests.exceptions.ConnectTimeout as ex:
            logger.error( f"[gTTS] timeout")
        except Exception as ex:
            logger.error( f"[gTTS] {ex}")
            logger.exception('')
        self._disable_gtts = time.time()
        return None,None

    def get_client(self):
        return OpenAI()

    def _text_to_audio_by_openai(self, text: str, emotion:int = 0) -> bytes:
        if self._disable_openai>0 and (time.time()-self._disable_openai)<180.0:
            return None,None
        try:
            vc:str = "alloy"
            if self.speaker==1001:
                vc = "alloy"
            elif self.speaker==1002:
                vc = "echo"
            elif self.speaker==1003:
                vc = "fable"
            elif self.speaker==1004:
                vc = "onyx"
            elif self.speaker==1005:
                vc = "nova"
            elif self.speaker==1006:
                vc = "shimmer"
            self._disable_openai = 0
            client:OpenAI = self.get_client()
            response:openai._base_client.HttpxBinaryResponseContent = client.audio.speech.create(
                model="tts-1",
                voice=vc,
                response_format="mp3",
                input=TtsEngine.__penpenpen(text,' ')
            )
            # openaiはmp3で返ってくる
            return response.content,f"OpenAI:{vc}"
        except requests.exceptions.ConnectTimeout as ex:
            logger.error( f"[gTTS] timeout")
        except Exception as ex:
            logger.error( f"[gTTS] {ex}")
            logger.exception('')
        self._disable_openai = time.time()
        return None,None

    @staticmethod
    def convert_blank( text:str ) ->str:
        text = re.sub( r'[「」・、。]+',' ',text)
        return text.strip()

    def _text_to_audio( self, text1: str, emotion:int = 0 ) -> bytes:
        if TtsEngine.EOT==text1:
            return self.sound_listen_out,''
        wave: bytes = None
        model:str = None
        text:str = TtsEngine.convert_blank( text1 )
        if 0<=self.speaker and self.speaker<1000:
            wave, model = self._text_to_audio_by_voicevox( text, emotion )
        if 1000<=self.speaker and self.speaker<2000:
            wave, model = self._text_to_audio_by_openai( text, emotion )
        if wave is None:
            wave, model = self._text_to_audio_by_gtts( text, emotion )
        return wave,model
        
    def run_talk(self)->None:
        start:bool = False
        while True:
            talk_id:int = -1
            text:str = None
            emotion: int = 0
            audio:bytes = None
            tts_model:str = None
            with self.lock:
                try:
                    talk_id, text, emotion, audio, tts_model = self.play_queue.get_nowait()
                except Exception as ex:
                    if not isinstance( ex, Empty ):
                        logger.exception('')
                    talk_id=-1
                    text = None
                    audio = None
                if text is None:
                    self._running_future2 = None
                    # 再生終了通知
                    if self.start_call is not None:
                        self.start_call( None, emotion, tts_model )
                    return
            try:
                if talk_id == self._talk_id:
                    # 再生開始通知
                    if self.start_call is not None:
                        self.start_call( text, emotion, tts_model )
                    # 再生処理
                    if audio is not None:
                        self._sound_init()
                        audio_buffer = BytesIO(audio)
                        audio_buffer.seek(0)
                        if (time.time()-self._last_talk)>5.0:
                            feed_buffer = BytesIO(self.feed_wave)
                            feed_buffer.seek(0)
                            pygame.mixer.music.load(feed_buffer)
                            pygame.mixer.music.play()
                            pygame.mixer.music.queue(audio_buffer)
                        else:
                            pygame.mixer.music.load(audio_buffer)
                            pygame.mixer.music.play()
                        while not pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        self._last_talk = time.time()
                    # 再生終了待ち
                    if audio is not None:
                        if pygame.mixer.music.get_busy():
                            while pygame.mixer.music.get_busy():
                                if talk_id != self._talk_id:
                                    pygame.mixer.music.stop()
                                    break
                                time.sleep(0.2)
                            time.sleep(0.5)
                        self._last_talk = time.time()
                    
            except Exception as ex:
                logger.exception('')

    def play_listn_in(self):
        self._play_beep( self.sound_listen_in )

    def play_listen_out(self):
        self._play_beep( self.sound_listen_out )

    def play_error1(self):
        self._play_beep( self.sound_error1 )

    def play_error2(self):
        self._play_beep( self.sound_error2 )

    def _play_beep(self, snd ):
        try:
            self._sound_init()
            if self.beep_ch is not None:
                while self.beep_ch.get_busy():
                    pygame.time.delay(200)
            wb: BytesIO = BytesIO( snd )
            wb.seek(0)
            sound = pygame.mixer.Sound( wb )
            self.beep_ch:pygame.mixer.Channel = sound.play(fade_ms=0)
            #pygame.time.delay( int(duratin_sec * 1000) )
            self._last_talk = time.time()
        except:
            logger.exception('')
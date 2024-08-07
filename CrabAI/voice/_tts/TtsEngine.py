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
from CrabAI.voice.voice_state import VoiceState

from logging import getLogger
logger = getLogger(__name__)

def _to_int( value, default:int ) -> int:
    try:
        val = int(value)
        if val>0:
            return val
    except:
        pass
    return default

class TtsEngine:
    EOT:str = "<|EOT|>"
    # 男性はM 女性はF
    VoiceList:list[tuple[str,int,str,str]] = [
        ( "VOICEVOX:四国めたん[ノーマル]",  2, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[あまあま]",  0, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[ツンツン]",  6, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[セクシー]",  4, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[ささやき]", 36, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[ヒソヒソ]", 37, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ノーマル]",  3, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[あまあま]",  1, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ツンツン]",  7, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[セクシー]",  5, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ささやき]", 22, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ヒソヒソ]", 38, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ヘロヘロ]", 75, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[なみだめ]", 76, 'ja_JP', 'F' ),
        ( "VOICEVOX:春日部つむぎ[ノーマル]",  8, 'ja_JP', 'F' ),
        ( "VOICEVOX:雨晴はう[ノーマル]",  10, 'ja_JP', 'F' ),
        ( "VOICEVOX:波音リツ[ノーマル]",   9, 'ja_JP', 'F' ),
        ( "VOICEVOX:波音リツ[クイーン]",  65, 'ja_JP', 'F' ),
        ( "VOICEVOX:玄野武宏[ノーマル]",  11, 'ja_JP', 'M' ),
        ( "VOICEVOX:玄野武宏[喜び]",     39, 'ja_JP', 'M' ),
        ( "VOICEVOX:玄野武宏[ツンギレ]",  40, 'ja_JP', 'M' ),
        ( "VOICEVOX:玄野武宏[悲しみ]",   41, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[ふつう]",  12, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[わーい]",  32, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[びくびく]", 33, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[おこ]",   34, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[びえーん]", 35, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[ノーマル]",  13, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[熱血]",    81, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[不機嫌]",   82, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[喜び]",    83, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[しっとり]",   84, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[かなしみ]",   85, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[囁き]",    86, 'ja_JP', 'M' ),
        ( "VOICEVOX:冥鳴ひまり[ノーマル]", 14, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[ノーマル]", 16, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[あまあま]", 15, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[ツンツン]", 18, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[セクシー]", 17, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[ささやき]", 19, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[ノーマル]", 20, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[セクシー／あん子]", 66, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[泣き]", 77, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[怒り]", 78, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[喜び]", 79, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[のんびり]", 80, 'ja_JP', 'F' ),
        ( "VOICEVOX:剣崎雌雄[ノーマル]", 21, 'ja_JP', 'M' ),
        ( "VOICEVOX:WhiteCUL[ノーマル]", 23, 'ja_JP', 'F' ),
        ( "VOICEVOX:WhiteCUL[たのしい]", 24, 'ja_JP', 'F' ),
        ( "VOICEVOX:WhiteCUL[かなしい]", 25, 'ja_JP', 'F' ),
        ( "VOICEVOX:WhiteCUL[びえーん]", 26, 'ja_JP', 'F' ),
        ( "VOICEVOX:後鬼[人間ver.]", 27, 'ja_JP', 'M' ),
        ( "VOICEVOX:後鬼[ぬいぐるみver.]", 28, 'ja_JP', 'M' ),
        ( "VOICEVOX:No.7[ノーマル]",  29, 'ja_JP', 'F' ),
        ( "VOICEVOX:No.7[アナウンス]", 30, 'ja_JP', 'F' ),
        ( "VOICEVOX:No.7[読み聞かせ]", 31, 'ja_JP', 'F' ),
        ( "VOICEVOX:ちび式じい[ノーマル]", 42, 'ja_JP', 'M' ),
        ( "VOICEVOX:櫻歌ミコ[ノーマル]", 43, 'ja_JP', 'F' ),
        ( "VOICEVOX:櫻歌ミコ[第二形態]", 44, 'ja_JP', 'F' ),
        ( "VOICEVOX:櫻歌ミコ[ロリ]",    45, 'ja_JP', 'F' ),
        ( "VOICEVOX:小夜/SAYO[ノーマル]", 46, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[ノーマル]", 47, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[楽々]", 48, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[恐怖]", 49, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[内緒話]", 50, 'ja_JP', 'F' ),
        ( "VOICEVOX:†聖騎士 紅桜†[ノーマル]", 51, 'ja_JP', 'M' ),
        ( "VOICEVOX:雀松朱司[ノーマル]", 52, 'ja_JP', 'M' ),
        ( "VOICEVOX:麒ヶ島宗麟[ノーマル]", 53, 'ja_JP', 'M' ),
        ( "VOICEVOX:春歌ナナ[ノーマル]", 54, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使アル[ノーマル]", 55, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使アル[おちつき]", 56, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使アル[うきうき]", 57, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使ビィ[ノーマル]", 58, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使ビィ[おちつき]", 59, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使ビィ[人見知り]", 60, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[ノーマル]", 61, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[おどろき]", 62, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[こわがり]", 63, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[へろへろ]", 64, 'ja_JP', 'F' ),
        ( "VOICEVOX:栗田まろん[ノーマル]", 67, 'ja_JP', 'F' ),
        ( "VOICEVOX:あいえるたん[ノーマル]", 68, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ノーマル]", 69, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[元気]", 70, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ささやき]", 71, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ぶりっ子]", 72, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ボーイ]", 73, 'ja_JP', 'F' ),
        ( "VOICEVOX:琴詠ニア[ノーマル]", 74, 'ja_JP', 'F' ),
        ( "OpenAI:alloy", 1001, 'ja_JP', 'X' ),
        ( "OpenAI:echo", 1002, 'ja_JP', 'X' ),
        ( "OpenAI:fable", 1003, 'ja_JP', 'X' ),
        ( "OpenAI:onyx", 1004, 'ja_JP', 'M' ), # 男性っぽい
        ( "OpenAI:nova", 1005, 'ja_JP', 'F' ), # 女性っぽい
        ( "OpenAI:shimmer", 1006, 'ja_JP', 'F' ), # 女性ぽい
        ( "gTTS:[ja_JP]", 2000, 'ja_JP', 'X' ),
        ( "gTTS:[en_US]", 2001, 'en_US', 'X' ),
        ( "gTTS:[en_GB]", 2002, 'en_GB', 'X' ),
        ( "gTTS:[fr_FR]", 2003, 'fr_FR', 'X' ),
    ]

    @staticmethod
    def id_to_model( idx:int ) -> tuple[str,int,str,str]|None:
        return next((voice for voice in TtsEngine.VoiceList if voice[1] == idx), None )

    @staticmethod
    def id_to_name( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        name = voice[0] if voice else None
        return name if name else '???'

    @staticmethod
    def id_to_lang( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        lang = voice[2] if voice else None
        return lang if lang else 'ja_JP'

    @staticmethod
    def id_to_gender( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        gender = voice[3] if voice else None
        return gender if gender else 'X'

    @staticmethod
    def load_default( conf:ShareParam ):
        pass

    def __init__(self, *, speaker:int=-1, submit_task = None, talk_callback = None, katakana_dir='tmp/katakana' ):
        # 並列処理用
        self.lock:Condition = Condition()
        self._audio_future:Future|None = None
        self._talk_future:Future|None = None
        self.wave_queue:Queue = Queue()
        self.play_queue:Queue = Queue()
        self._last_talk:float = 0
        self._stop_delay:float = 0.2
        self._power_timeout:float = 5.0
        # 発声中のセリフのID
        self._talk_id: int = 0
        self._talk_seq: int = 0
        # 音声エンジン選択
        self.speaker = speaker
        # コールバック
        self.executor:ThreadPoolExecutor|None = None
        self._submit_call = submit_task # スレッドプールへの投入
        self._start_call = talk_callback # 発声開始と完了を通知する
        # pygame初期化済みフラグ
        self.pygame_init:bool = False
        # beep
        self.beep_ch:list[pygame.mixer.Channel]  = []
        # 音声エンジン無効時間
        self._disable_gtts: float = 0.0
        self._disable_openai: float = 0.0
        self._disable_voicevox: float = 0.0
        # VOICEVOXサーバURL
        self._voicevox_url:str|None = None
        self._voicevox_port = _to_int( os.getenv('VOICEVOX_PORT' ), 50021)
        self._voicevox_list = list(set([os.getenv('VOICEVOX_HOST','127.0.0.1'),'127.0.0.1','192.168.0.104','chickennanban.ddns.net','chickennanban1.ddns.net','chickennanban2.ddns.net','chickennanban3.ddns.net']))
        self._katakana_dir = katakana_dir

        # フェードイン用
        self.feed = create_tone( 32, time=0.4, volume=0.9, sample_rate=16000)
        self.feed = 0.5 * np.sin( np.linspace(0, np.pi, int(16000*0.4)) ) # np.linspace(0.5, 0.0, num=int(16000*0.4) )
        self.feed_wave = audio_to_wave_bytes(self.feed, sample_rate=16000 )
        # 再生完了を同期するためのデータ
        self.sync_wave = audio_to_wave_bytes(np.zeros(1, dtype=np.float32), sample_rate=16000 )
        # 通知用の音声データ
        self.sound_mute_in = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 ce", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_mute_out = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 ec", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_listen_in = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 ee", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_listen_out = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 cc", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_talk_end = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v10 eg", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_error1 = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t240v15 O3aa", sampling_rate=16000 ))), sample_rate=16000 )
        self.sound_error2 = audio_to_wave_bytes( np.concatenate((self.feed,mml_to_audio( "t480v15 O3aaa", sampling_rate=16000 ))), sample_rate=16000 )

    def _fn_callback(self, stat:int, talk_id:int, seq:int, content:str|None, emotion:int, model:str|None):
        if self._start_call is not None:
            self._start_call( stat, talk_id, seq, content, emotion, model )

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
                logger.info(f"[PyGame]reset")
                pygame.mixer.quit()
        except:
            logger.exception("can not reset pygame")
        try:
            if not self.pygame_init:
                logger.info(f"[PyGame]init")
                #pygame.mixer.pre_init(16000,-16,1,10240)
                pygame.mixer.quit()
                pygame.mixer.init(16000,-16,1,4096)
                self.pygame_init = True
        except:
            logger.exception("can not reset pygame")
        # if not self.pygame_init:
        #     pygame.init()
        #     pygame.mixer.pre_init(16000,-16,1,10240)
        #     pygame.mixer.quit()
        #     pygame.mixer.init()
        #     self.pygame_init = True

    def _fn_submit_task(self, func ) -> Future:
        if self._submit_call is not None:
            return self._submit_call(func)
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=4)
        return self.executor.submit( func )

    def cancel(self):
        self._talk_id += 1
        self._talk_seq = 0

    def is_playing(self) ->bool:
        if not self.wave_queue.empty() or not self.play_queue.empty():
            return True
        if self._audio_future is not None or self._talk_future is not None:
            return True
        return False

    def _get_voicevox_url( self ) ->str|None:
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
        # テキストを行単位に分割する
        lines:list[str] = TtsEngine.split_talk_text(full_text)
        # 最初の再生の時だけ、先頭の単語を短くする
        if not self.is_playing():
            firstline:str = lines[0]
            p:int = firstline.find('、')
            if 1<p and p<len(firstline)-1:
                lines[0] = firstline[:p+1]
                lines.insert(1, firstline[p+1:])
        # キューに追加する
        talk_id:int = self._talk_id
        for text in lines:
            logger.info(f"[TTS] wave_queue.put {text}")
            self.wave_queue.put( (talk_id, self._talk_seq, text, emotion ) )
            self._fn_callback( VoiceState.ST_TALK_ENTRY, talk_id, self._talk_seq, text, emotion, None )
            self._talk_seq += 1
        # 処理スレッドを起動する
        with self.lock:
            if self._audio_future is None:
                self._audio_future = self._fn_submit_task(self._th_run_text_to_audio)
    
    def _th_run_text_to_audio(self)->None:
        """ボイススレッド
        テキストキューからテキストを取得して音声に変換して発声キューへ送る
        """
        logger.info(f"[TTS] text_to_audio thread start")
        last_time:float = time.time()
        while True:
            talk_id:int = -1
            text:str|None = None
            emotion:int = -1
            with self.lock:
                try:
                    talk_id, seq, text, emotion = self.wave_queue.get_nowait()
                    last_time = time.time()
                except Exception as ex:
                    if not isinstance( ex, Empty ):
                        logger.exception(ex)
                    talk_id=-1
                    text = None
                if text is None and (time.time()-last_time)>2.0:
                    self._audio_future = None
                    logger.info(f"[TTS] text_to_audio thread end")
                    return
            if text is None:
                time.sleep(0.2)
                continue
            try:
                if talk_id == self._talk_id: # cancelされてなければ
                    self._music_wakeup()
                    logger.debug(f"[TTS] text_to_audio {text}")
                    # textから音声へ
                    self._fn_callback( VoiceState.ST_TALK_CONVERT_START, talk_id, seq, text, emotion, None )
                    audio_bytes, tts_model = self._text_to_audio( text, emotion )
                    self._fn_callback( VoiceState.ST_TALK_CONVERT_END, talk_id, seq, text, emotion, tts_model )
                    self.play_queue.put( (talk_id,seq,text,emotion,audio_bytes,tts_model) )
                    with self.lock:
                        if self._talk_future is None:
                            logger.info("[TTS] play thread submit")
                            self._talk_future = self._fn_submit_task(self._th_run_talk)
                        else:
                            logger.info("[TTS] play thread running")
            except Exception as ex:
                logger.exception(ex)

    @staticmethod
    def __penpenpen( text, default=" " ) ->str:
        if text is None or text.startswith("```"):
            return default # VOICEVOX,OpenAI,gTTSで、エラーにならない無音文字列
        else:
            return text
        
    def _text_to_audio_by_voicevox(self, text: str, emotion:int = 0, lang='ja') -> tuple[bytes|None,str|None]:
        if self._disable_voicevox>0 and (time.time()-self._disable_voicevox)<180.0:
            return None,None
        sv_url: str|None = self._get_voicevox_url()
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
            res1_json['speedScale'] = ss*1.1
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
            if res.status_code == 200:
                model:str = TtsEngine.id_to_name(self.speaker)
                # wave形式 デフォルトは24kHz
                return res.content, model
            logger.error( f"[VOICEVOX] code:{res.status_code} {res.text}")
        except requests.exceptions.ConnectTimeout as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
        except requests.exceptions.ConnectionError as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
        except Exception as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
            logger.exception('')
        self._disable_voicevox = time.time()
        return None,None

    def _text_to_audio_by_gtts(self, text: str, emotion:int = 0) -> tuple[bytes|None,str|None]:
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

    def _text_to_audio_by_openai(self, text: str, emotion:int = 0) -> tuple[bytes|None,str|None]:
        if self._disable_openai>0 and (time.time()-self._disable_openai)<180.0:
            return None,None
        try:
            model_name:str = TtsEngine.id_to_name(self.speaker)
            voice_model:str = model_name[len("OpenAI:"):] if model_name is not None and model_name.startswith("OpenAI:") else ""
            if not voice_model:
                return None,None
            self._disable_openai = 0
            client:OpenAI = self.get_client()
            response:openai._base_client.HttpxBinaryResponseContent = client.audio.speech.create(
                model="tts-1",
                voice=voice_model, # type: ignore
                response_format="mp3",
                input=TtsEngine.__penpenpen(text,' ')
            )
            # openaiはmp3で返ってくる
            return response.content, model_name
        except requests.exceptions.ConnectTimeout as ex:
            logger.error( f"[OpenAI] timeout")
        except Exception as ex:
            logger.exception(f"[OpenAI] {ex}")
        self._disable_openai = time.time()
        return None,None

    @staticmethod
    def convert_blank( text:str ) ->str:
        text = re.sub( r'[「」・、。]+',' ',text)
        return text.strip()

    def _text_to_audio( self, text1: str, emotion:int = 0 ) -> tuple[bytes|None,str|None]:
        if TtsEngine.EOT==text1:
            return self.sound_mute_out,''
        wave: bytes|None = None
        model:str|None = None
        text:str = TtsEngine.convert_blank( text1 )
        if 0<=self.speaker and self.speaker<1000:
            wave, model = self._text_to_audio_by_voicevox( text, emotion )
        if 1000<=self.speaker and self.speaker<2000:
            wave, model = self._text_to_audio_by_openai( text, emotion )
        if wave is None:
            wave, model = self._text_to_audio_by_gtts( text, emotion )
        return wave,model

    TALK_RUN:int = 0
    TALK_FLUSH:int = 1
    TALK_END_WAIT:int = 2
    TALK_END:int = 3
    def _th_run_talk(self)->None:
        logger.info(f"[TTS] play thread start")
        status:int = TtsEngine.TALK_RUN
        while True:
            talk_id:int = -1
            text:str|None = None
            emotion: int = 0
            audio:bytes|None = None
            tts_model:str|None = None
            with self.lock:
                try:
                    talk_id, seq, text, emotion, audio, tts_model = self.play_queue.get_nowait() # lockの中だからnowaitにする
                    status = TtsEngine.TALK_RUN
                except Exception as ex:
                    if not isinstance( ex, Empty ):
                        logger.exception('') # Empty以外のエラーならログ
                    talk_id=-1
                    text = None
                    audio = None
                    if not self.wave_queue.empty() or self._audio_future is not None:
                        # キューは空だったが音声化処理が動いている
                        status = TtsEngine.TALK_RUN
                    elif not pygame.mixer.music.get_busy():
                        # キューは空だった。音声再生も終了しているが、騙されてはいけない。
                        # get_busy()がFalseになっても、ほんのしばらく再生は続いている。
                        if status == TtsEngine.TALK_RUN:
                            status = TtsEngine.TALK_FLUSH # 駄目押しの無音再生
                        elif status == TtsEngine.TALK_END_WAIT:
                            # 無音再生が完了して規定時間経過後に終了と見做す
                            if (time.time()-self._last_talk) > self._stop_delay:
                                self._talk_future = None
                                status = TtsEngine.TALK_END # 完全に終わった
                    else:
                        # 再生中なら時間を更新しておく
                        self._last_talk = time.time()

            if status == TtsEngine.TALK_FLUSH:
                # 駄目押しの無音再生
                logger.info(f"[TTS] play {talk_id} empty")
                # buffer_talk_end = BytesIO(self.sound_talk_end)
                # buffer_talk_end.seek(0)
                buffer_sync = BytesIO(self.sync_wave)
                buffer_sync.seek(0)
                # pygame.mixer.music.load(buffer_talk_end)
                # pygame.mixer.music.play() #
                # pygame.mixer.music.queue(buffer_sync,loops=2)
                pygame.mixer.music.load(buffer_sync)
                pygame.mixer.music.play() #
                self._last_talk = time.time()
                status = TtsEngine.TALK_END_WAIT
                time.sleep(0.2)
                continue
            elif status == TtsEngine.TALK_END_WAIT:
                # 無音再生完了まち
                time.sleep(0.2)
                continue
            elif status == TtsEngine.TALK_END:
                # 完全に終わった
                logger.info(f"[TTS] play thread exit")
                # 再生終了通知
                self._fn_callback( VoiceState.ST_TALK_EXIT, talk_id, -1, None, emotion, tts_model )
                return
            else:
                if talk_id<0 and text is None and audio and None:
                    time.sleep(0.2)
                    continue
                elif talk_id != self._talk_id: # cancelされた
                    logger.info(f"[TTS] play {status} {talk_id} {seq} cancel")
                    time.sleep(0.01)
                    continue

            try:
                # 再生開始通知
                self._fn_callback( VoiceState.ST_TALK_PLAY_START, talk_id, seq, text, emotion, tts_model )
                # 再生処理
                if audio is not None:
                    self._sound_init()
                    self._music_wakeup()
                    audio_buffer = BytesIO(audio)
                    audio_buffer.seek(0)
                    if not pygame.mixer.music.get_busy():
                        # 再生中でなければ、ロードして再生開始
                        logger.info(f"[TTS] play {talk_id} {seq} start {text}")
                        pygame.mixer.music.load(audio_buffer)
                        pygame.mixer.music.play()
                    else:
                        # 前の音声が再生中なのでキューにいれる
                        logger.info(f"[TTS] play {talk_id} {seq} queue {text}")
                        pygame.mixer.music.queue(audio_buffer)
                    self._last_talk = time.time()
                    # 再生が完了するまで待機
                    # おそらくバッファの関係で、再生完了の0.5秒前くらいにget_busy()がFalseになる
                    while pygame.mixer.music.get_busy():
                        self._last_talk = time.time()
                        if talk_id != self._talk_id: # キャンセルされた
                            pygame.mixer.music.stop()
                            logger.info(f"[TTS] play {talk_id} {seq} cancel")
                            break
                        time.sleep(0.2)
                # 再生完了通知
                self._fn_callback( VoiceState.ST_TALK_PLAY_END, talk_id, seq, text, emotion, tts_model )

            except Exception as ex:
                logger.exception('')

    def play_mute_in(self):
        logger.info("beep mute_in")
        self._play_beep( self.sound_mute_in )

    def play_mute_out(self, *, wait=False):
        logger.info("beep mute_out")
        self._play_beep( self.sound_mute_out, wait=wait )

    def play_listen_in(self, *, wait=False):
        logger.info("beep listen_in")
        self._play_beep( self.sound_listen_in, wait=wait )

    def play_listen_out(self):
        logger.info("beep listen_out")
        self._play_beep( self.sound_listen_out )

    def play_talk_end(self):
        logger.info("beep talk_end")
        self._play_beep( self.sound_talk_end, wait=True )

    def play_error1(self):
        logger.info("beep error1")
        self._play_beep( self.sound_error1 )

    def play_error2(self):
        logger.info("beep error2")
        self._play_beep( self.sound_error2 )

    def _play_beep(self, snd, *, wait=False ):
        try:
            self._sound_init()
            while self.beep_ch:
                ch:pygame.mixer.Channel = self.beep_ch.pop(0)
                while ch.get_busy():
                    pygame.time.delay(200)
            wb: BytesIO = BytesIO( snd )
            wb.seek(0)
            sound = pygame.mixer.Sound( wb )
            self.beep_ch.append( sound.play(fade_ms=0) )
            #pygame.time.delay( int(duratin_sec * 1000) )
            self._last_talk = time.time()
            if wait:
                while self.beep_ch:
                    ch:pygame.mixer.Channel = self.beep_ch.pop(0)
                    while ch.get_busy():
                        pygame.time.delay(200)
                pygame.time.delay(200)
            self._last_talk = time.time()
        except:
            logger.exception('')

    def _music_wakeup(self):
        try:
            # 最後の再生から一定時間すぎると、OSのオーディオがPowerOffになるみたい。
            # この状態から再生を始めると、最初の音がフェードインするので、最初の言葉が聞き取れない
            # だから、最初にダミーの音を再生してフェードイン中になにか再生する
            self._sound_init()
            if pygame.mixer.music.get_busy():
                self._last_talk = time.time()
            elif (time.time()-self._last_talk)>self._power_timeout:
                logger.info(f"[TTS] wakeup")
                feed_buffer = BytesIO(self.feed_wave)
                feed_buffer.seek(0)
                pygame.mixer.music.load(feed_buffer)
                pygame.mixer.music.play()
                self._last_talk = time.time()
        except:
            logger.exception('')
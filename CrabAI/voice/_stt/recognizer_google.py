import sys,os,traceback,time,json
import numpy as np
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from speech_recognition.audio import AudioData
#from .VoskUtil import sound_float_to_int16, NetworkError

import logging
logger = logging.getLogger('voice')

def to_int( value, default ):
    return int(value) if isinstance(value,(int,float)) else int(default)
def to_float( value, default ):
    return float(value) if isinstance(value,(int,float)) else float(default)

class RecognizerGoogle:

    def __init__(self, timeout=None, sample_rate:int=16000, width=2, lang='ja_JP'):
        self.operation_timeout = timeout
        self.sample_rate:int = sample_rate
        self.sample_width = width
        self.lang = lang

    def stop(self):
        pass

    # def recognizef(self, float_list:list[float], *,timeout=None, retry=None, sample_rate:int=None, lang:str=None  ):
    #     """音声認識 float配列バージョン"""
    #     floats = np.array(float_list, dtype=np.float32)
    #     #intdata = np.int16( floats * 32767 )
    #     intdata = sound_float_to_int16( floats, scale=0.8, lowcut=0 )
    #     bytes_data = intdata.tobytes()
    #     if sample_rate is None:
    #         sample_rate = self.sample_rate
    #     audio_data = AudioData( bytes_data, sample_rate, 2 )
    #     return self._recognize_audiodata( audio_data, timeout=timeout, lang=lang )

    # def recognizeb(self, buf:bytes, *, timeout=None, retry=None, sample_rate:int=None, sample_width:int = None, lang:str=None ):
    #     if sample_rate is None:
    #         sample_rate = self.sample_rate
    #     if sample_width is None:
    #         sample_width = self.sample_width
    #     lang = self.lang
    #     audio_data = AudioData( buf, sample_rate, sample_width)
    #     return self._recognize_audiodata( audio_data, lang=lang )

    @staticmethod        
    def recognize( audio_data:AudioData, *, sample_rate:int=None, timeout:float=None, retry:int=None, lang:str=None ):
        if audio_data is None:
            return '',1.0
        elif isinstance(audio_data,AudioData):
            sample_rate = audio_data.sample_rate
        elif isinstance(audio_data,np.ndarray):
            sample_rate = to_int(sample_rate,16000)
            if audio_data.dtype==np.float32:
                f = audio_data * 32767.0
                audio_data = AudioData( f.astype(np.int16).tobytes(), sample_rate, 2 )
            elif audio_data.dtype==np.int16:
                audio_data = AudioData( audio_data.tobytes(), sample_rate, 2 )
            else:
                raise TypeError( f"invalid audio {audio_data.dtype}")
        else:
            raise TypeError( f"invalid audio type {type(audio_data)}")

        if not isinstance(lang,str) or len(lang)<5:
            lang = 'ja_JP'
        retry = int(retry) if isinstance(retry,(int,float)) and 0<=retry else 3
        timeout = float(timeout) if isinstance(timeout,(int,float)) and timeout>=1 else 15.0

        try:
            for trycount in range(0, retry+1):
                try:
                    actual_result = RecognizerGoogle._recognize_google(audio_data, language=lang, operation_timeout=timeout )
                except (HTTPError,TimeoutError) as ex:
                    if trycount==retry:
                        raise ex
                    logger.debug(f"[RECG] try{trycount} error response {ex}")
                    continue
                except URLError as ex:
                    logger.debug(f"[RECG] try{trycount} error response {ex}")
                    raise ex
                break

            if not actual_result or not isinstance( actual_result, dict ):
                logger.debug(f"[RECG] abort or no result {type(actual_result)}")
                return None,None
            aaaa = actual_result.get("alternative", []) or []
            if len(aaaa)==0 or not actual_result.get('final',False): 
                data=json.dumps( actual_result, indent=2, ensure_ascii=False )
                logger.error(f"[RECG] error response {actual_result}")
                logger.error(f"ERROR:actual_result:{data}")
                return None,None
            if "confidence" in aaaa:
                # return alternative with highest confidence score
                best_hypothesis = max(aaaa, key=lambda alternative: alternative["confidence"])
            else:
                # when there is no confidence available, we arbitrarily choose the first hypothesis.
                best_hypothesis = aaaa[0]
            if "transcript" in best_hypothesis:
                # https://cloud.google.com/speech-to-text/docs/basics#confidence-values
                # "Your code should not require the confidence field as it is not guaranteed to be accurate, or even set, in any of the results."
                confidence = best_hypothesis.get("confidence", 1.0) or 0.0
                final_text = best_hypothesis.get("transcript") or ''
                logger.debug(f"[RECG] {confidence} {final_text}")
                return final_text, confidence

        except (HTTPError,URLError,TimeoutError) as ex:
            raise ex
        except Exception as ex:
            logger.exception('')
            raise ex
        return None,None

    @staticmethod
    def get_recognizers_key():
        VAR='RECOGNIZERS_GOOGLE_KEY'
        key = os.environ.get(VAR)
        if key is None:
            key = ''
            try:
                import re
                import inspect
                import speech_recognition.recognizers.google
                filepath=inspect.getfile(speech_recognition.recognizers.google)
                pattern = r'^\s*key\s*=\s*"([^"]*)"'
                with open( filepath, 'r', encoding='utf-8') as file:
                    for line in file:
                        match = re.match(pattern, line)
                        if match:
                            # 二重引用符で囲まれた内容を取り出し
                            value = match.group(1)
                            if isinstance(value,str) and len(value)>0:
                                key = value
                                break
            except:
                pass
            os.environ[VAR] = key
        return key

    @staticmethod
    def _recognize_google( audio_data, key=None, language="en-US", pfilter=0, operation_timeout=None):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Google Speech Recognition API.
        ``audio_data``（``AudioData``インスタンス）に対して音声認識を行い、Google音声認識APIを使用します。

        The Google Speech Recognition API key is specified by ``key``. If not specified, it uses a generic key that works out of the box. 
        This should generally be used for personal or testing purposes only, as it **may be revoked by Google at any time**.

        Google音声認識APIキーは、``key``によって指定されます。指定されていない場合、箱から出してすぐに機能する一般的なキーを使用します。
        これは一般的に個人的な目的やテスト目的でのみ使用すべきであり、**いつでもGoogleによって取り消される可能性があります**。

        To obtain your own API key, simply following the steps on the `API Keys <http://www.chromium.org/developers/how-tos/api-keys>`__ page
        at the Chromium Developers site. In the Google Developers Console, Google Speech Recognition is listed as "Speech API".

        独自のAPIキーを取得するには、Chromium Developersサイトの`APIキー<http://www.chromium.org/developers/how-tos/api-keys>`__ ページの
        手順に従ってください。Google Developers Consoleでは、Google音声認識は「Speech API」としてリストされています。

        The recognition language is determined by ``language``, an RFC5646 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), 
        defaulting to US English. A list of supported language tags can be found in this `StackOverflow answer <http://stackoverflow.com/a/14302134>`__.

        認識言語は、``"en-US"``（アメリカ英語）や``"fr-FR"``（フランス語国際）など、RFC5646言語タグによって``language``で決定され、デフォルトはアメリカ英語です。
        対応する言語タグのリストは、この`StackOverflowの回答<http://stackoverflow.com/a/14302134>`__ で見つかります。

        The profanity filter level can be adjusted with ``pfilter``: 0 - No filter, 1 - Only shows the first character and replaces the rest with asterisks.
        The default is level 0.

        不適切な言葉のフィルターレベルは、``pfilter``で調整できます: 0 - フィルターなし、1 - 最初の文字のみを表示して残りをアスタリスクで置き換えます。
        デフォルトはレベル0です。

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible.
        Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
        assert key is None or isinstance(key, str), "``key`` must be ``None`` or a string"
        assert isinstance(language, str), "``language`` must be a string"

        if not isinstance(operation_timeout,float) or operation_timeout<1.0:
            operation_timeout = 15

        flac_data = audio_data.get_flac_data(
            convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
            convert_width=2  # audio samples must be 16-bit
        )
        if key is None: key = RecognizerGoogle.get_recognizers_key()
        url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
            "client": "chromium",
            "lang": language,
            "key": key,
            "pFilter": pfilter
        }))
        try:
            request = Request(url, data=flac_data, headers={"Content-Type": "audio/x-flac; rate={}".format(audio_data.sample_rate)})
            with urlopen(request, timeout=operation_timeout) as response:
                response_text = response.read().decode("utf-8")
        except (HTTPError,TimeoutError,URLError) as ex:
            logger.error( f"google recognize exception: {ex}")
            raise ex

        logger.debug( f"google recognize response: {response.status} {response_text}")

        # ignore any blank blocks
        actual_result = []
        count:int = 0
        for line in response_text.split("\n"):
            if not line: continue
            data = json.loads(line)
            result = data.get("result")
            if len(result) != 0:
                actual_result = result[0]
                break

        # return results
        return actual_result

        # if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()

        # if "confidence" in actual_result["alternative"]:
        #     # return alternative with highest confidence score
        #     best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
        # else:
        #     # when there is no confidence available, we arbitrarily choose the first hypothesis.
        #     best_hypothesis = actual_result["alternative"][0]
        # if "transcript" not in best_hypothesis: raise UnknownValueError()
        # # https://cloud.google.com/speech-to-text/docs/basics#confidence-values
        # # "Your code should not require the confidence field as it is not guaranteed to be accurate, or even set, in any of the results."
        # confidence = best_hypothesis.get("confidence", 0.5)
        # if with_confidence:
        #     return best_hypothesis["transcript"], confidence
        # return best_hypothesis["transcript"]


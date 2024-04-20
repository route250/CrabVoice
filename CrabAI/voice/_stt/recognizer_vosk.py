import sys
import os
from pathlib import Path
import re
import json
import logging

try:
    import vosk
    from vosk import Model, SpkModel, KaldiRecognizer
except:
    pass

logger = logging.getLogger('voice')

def get_vosk_model( lang:str='ja' ) ->Model:
    # search and load model
    for pattern in [ rf"vosk-model-{lang}",rf"vosk-model-small-{lang}"]:
        for directory in vosk.MODEL_DIRS:
            if directory is None or not Path(directory).exists():
                continue
            model_file_list = [ f for f in os.listdir(directory) if os.path.isdir(f) ]
            model_file = [model for model in model_file_list if re.match(pattern, model)]
            if len(model_file) == 0:
                continue
            try:
                return Model(str(Path(directory, model_file[0])))
            except:
                logger.exception(f"can not load vosk model {model_file[0]}")

    # cleanup zip file when download error?
    try:
        for directory in vosk.MODEL_DIRS:
            if directory is None or not Path(directory).exists():
                continue
            for f in os.listdir(directory):
                ff = os.path.join(directory,f)
                if os.path.isfile(ff) and re.match(r"vosk-model(-small)?-{}.*\.zip".format(lang), f):
                    logger.error(f"remove vosk model {ff}")
                    os.unlink(ff)
    except Exception as ex:
        logger.exception('cleanup??')

    # download model
    try:
        m:Model = Model(lang=lang)
        return m
    except Exception as ex:
        logger.exception(f"ERROR:can not load vosk model {ex}")
    return None

def get_vosk_spk_model(m:Model=None):
    for directory in vosk.MODEL_DIRS:
        if directory is None or not Path(directory).exists():
            continue
        model_file_list = os.listdir(directory)
        model_file = [model for model in model_file_list if re.match(r"vosk-model-spk-", model)]
        if model_file != []:
            return SpkModel(str(Path(directory, model_file[0])))
        
    p:str = m.get_model_path('vosk-model-spk-0.4',None) if m is not None else None
    if p is not None:
        return SpkModel( p )
    return None

katakana_basic = [
    # 基本のカタカナ50音
    "ア", "イ", "ウ", "エ", "オ",
    "カ", "キ", "ク", "ケ", "コ",
    "サ", "シ", "ス", "セ", "ソ",
    "タ", "チ", "ツ", "テ", "ト",
    "ナ", "ニ", "ヌ", "ネ", "ノ",
    "ハ", "ヒ", "フ", "ヘ", "ホ",
    "マ", "ミ", "ム", "メ", "モ",
    "ヤ", "ユ", "ヨ",
    "ラ", "リ", "ル", "レ", "ロ",
    "ワ", "ヲ", "ン",
    # 濁音
    "ガ", "ギ", "グ", "ゲ", "ゴ",
    "ザ", "ジ", "ズ", "ゼ", "ゾ",
    "ダ", "ヂ", "ヅ", "デ", "ド",
    "バ", "ビ", "ブ", "ベ", "ボ",
    # 半濁音
    "パ", "ピ", "プ", "ペ", "ポ",
]

def is_all_katakana(char):
    """文字がカタカナかどうかを判定する"""
    return ord(char) in range(0x30A1, 0x30FC + 1)

def is_small_katakana(char):
    return char in "ァィゥェォッャュョヮヵヶ・ー"

def is_katakana(char):
    return not is_small_katakana(char) and is_all_katakana(char)

def get_katakana_grammar():
    mdl:Model = get_vosk_model()
    model_path=mdl.get_model_by_name('vosk-model-small-ja-0.22')
    words_path=os.path.join(model_path, 'graph','words.txt' )
    word_set = set()
    for cc in katakana_basic:
        word_set.add(cc)
    with open(words_path,'r') as stream:
        while True:
            line = stream.readline()
            if line == '':
                break
            word = line.split()[0]
            # 先頭がカタカナ以外なら無視
            if not is_katakana(word[0]):
                continue
            katakana_len = 0
            for cc in word:
                if not is_all_katakana(cc):
                    # カタカナ以外の文字があれば無視
                    katakana_len = 0
                    break
                else:
                    if not is_small_katakana(cc):
                        katakana_len += 1
            if 0<katakana_len and katakana_len<3:
                word_set.add(word)
    for cc in 'ンゾヂヅヌ':
        if cc in word_set:
            word_set.remove(cc)
    word_list = sorted(word_set)
    grammers=json.dumps(word_list,ensure_ascii=False)
    return grammers
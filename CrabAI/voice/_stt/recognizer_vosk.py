import sys
import os
from pathlib import Path
import re
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

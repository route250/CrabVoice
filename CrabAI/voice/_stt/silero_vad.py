import os,sys,traceback
from logging import getLogger

import numpy as np

_exists_torch:bool = False
try:
    import torch
    _exists_torch = True
except:
    pass

logger = getLogger(__name__)

def init_jit_model(model_path: str, device:str='cpu'):
    try:
        logger.info(f"load {model_path}")
        torch.set_grad_enabled(False)
        model = torch.jit.load(model_path, map_location=torch.device(device))
        model.eval()
        return model
    except:
        logger.exception(f"failled to load {model_path}")
        return None

class SileroVAD:

    def __init__(self, window_size_samples=None, sampling_rate=None, device:str='cpu'):
        if sampling_rate != 8000 and sampling_rate != 16000:
            raise AssertionError("Currently silero VAD models support 8000 and 16000 sample rates")
        if sampling_rate == 16000:
            if window_size_samples not in [512, 1024, 1536]:
                raise AssertionError("Supported window_size_samples: [512, 1024, 1536] for 16000 sampling_rate")
        else:
            if window_size_samples not in [256, 512, 768]:
                raise AssertionError("Supported window_size_samples: [256, 512, 768] for 8000 sampling_rate")
        self.window_size_samples:int = int(window_size_samples)
        self.sampling_rate:int = int(sampling_rate)
        self.device=device
        self._can_not_load:bool = False
        self.model=None

    @staticmethod
    def scan_model_file():
        home_dir = os.path.expanduser("~")
        cache_dir=os.path.join( home_dir, ".cache" )
        master_dir=os.path.join( cache_dir, "torch","hub", "snakers4_silero-vad_master" )
        files_dir=os.path.join( master_dir, "files" ) # 旧バージョンはここにある
        data_dir=os.path.join( master_dir, 'src','silero_vad','data' ) # 新しいバージョンはここにある
        for dir in [files_dir,data_dir]:
            model_path = os.path.join( dir, "silero_vad.jit")
            if os.path.exists(model_path):
                return model_path
        return None

    def load(self) ->None:
        try:
            if self.model is None and _exists_torch and not self._can_not_load:
                model_path = SileroVAD.scan_model_file()
                if model_path is None:
                    # download example
                    logger.info("download SileroVAD models")
                    #torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
                    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',trust_repo=True)
                    model_path = SileroVAD.scan_model_file()
                if model_path is None:
                    self._can_not_load = True
                    raise AssertionError("can not load model in .cache directory")
                self.model = init_jit_model(model_path, device=self.device)
        except Exception as ex:
            self._can_not_load = True
            logger.exception("can not load SileroVAD model")

    def reset_states(self) ->None:
        try:
            if self.model:
                self.model.reset_states()
        except:
            logger.exception("")

    def is_speech(self,audio:np.ndarray) ->float:
        if not isinstance(audio,np.ndarray) or len(audio.shape)!=1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")
        if audio.shape[0] != self.window_size_samples:
            raise ValueError("invalid window size")
        if audio.dtype != np.float32:
            raise ValueError("numpy dtype is not float32")
        try:
            audio_torch = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")
        try:
            if self.model:
                speech_prob = self.model(audio_torch, self.sampling_rate).item()
                return round( float(speech_prob), 3 )
        except:
            logger.exception("")
        return 0.0

    def is_speech_torch(self,audio) ->float:
        try:
            if self.model:
                speech_prob = self.model(audio, self.sampling_rate).item()
                return speech_prob
        except:
            logger.exception("")
        return 0.0

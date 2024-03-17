import sys
import os
from io import BytesIO
from threading import Thread, Condition
import time
import wave
import traceback
import numpy as np

import pygame
from voice_utils import RingBuffer, AudioRingBuffer, VadCounter, towave, load_wave

class Txx:

    def __init__(self):
        self.vad:VadCounter = VadCounter()
        self.fr=16000
        self.num_fr = 0
        self.dict_list:list[dict] = []
        self.audio:np.ndarray = None

    def audio_callback(self,data):
        start_time = self.num_fr / self.fr
        end_time = (self.num_fr+len(data))/self.fr
        left = data.reshape(-1) if data.shape[1]==1 else data[:,0]
        start_state,up_trigger,dn_trigger,end_state = self.vad.put_f32(left)
        if not start_state:
            if end_state:
                print( f"Up {start_time:.3f}(sec)")
            elif up_trigger:
                print( f"UpDn {start_time:.3f}(sec)")
        else:
            if not end_state:
                print( f"Dn {start_time:.3f}(sec)")
            elif dn_trigger:
                print( f"DnUp {start_time:.3f}(sec)")
        #
        if self.audio is None:
            if start_state or up_trigger or end_state:
                self.audio = data
                self.dict_list.append({ 'start': start_time, 'end':end_time, 'audio': self.audio })
        else:
            if start_state or up_trigger:
                self.audio = np.concatenate( (self.audio,data), axis=0 )
                item = self.dict_list[-1]
                item['audio'] = self.audio
                item['end'] = end_time
            if dn_trigger or not end_state:
                if end_state:
                    self.audio = data
                    self.dict_list.append({ 'start': start_time, 'end':end_time, 'audio': self.audio })
                else:
                    self.audio = None
        self.num_fr += len(data)

def main():
    wav_filename='testData/nakagawke01.wav'

    print( f"#Split audio")
    xxx:Txx = Txx()
    load_wave(wav_filename, callback=xxx.audio_callback, wait=False )

    from faster_whisper import WhisperModel


    print( f"#Load model")
    model_size = "large-v3"
    t0=time.time()
    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # or run on GPU with INT8
    model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print( f"#Start")
    for idx,item in enumerate(xxx.dict_list):
        s = item.get('start',-1)
        e = item.get('end',-1)
        audio = item.get('audio')
        wav = towave( audio )
        print( f"{idx:3d} {s:8.3f} - {e:8.3f}" )
        content:str = ''
        t1=time.time()
        segments, info = model.transcribe( wav, beam_size=5, language='ja' )
        print(segments)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            content += ' ' + segment.text
        t2=time.time()
        print( f"# TIME {t1-t0} {t2-t1}")
        item['content'] = content

    # Pygameの初期化
    pygame.init()
    pygame.mixer.init()
    n=-1
    while True:
        if n<0:
            for idx,item in enumerate(xxx.dict_list):
                s = item.get('start',-1)
                e = item.get('end',-1)
                txt = item.get('content','')
                print( f"{idx:3d} {s:8.3f} - {e:8.3f} : {txt}" )
        keyin = input()
        try:
            n = int(keyin)
        except:
            n=-1
        if 0<=n and n<len(xxx.dict_list):
            print(f"{n}")
            au = xxx.dict_list[n].get('audio')
            wav = towave( au )
            # オンメモリのwaveデータを読み込む
            wave_sound = pygame.mixer.Sound(wav)
            # 再生
            wave_sound.play()

if __name__ == "__main__":
    main()
import sys,os
from io import BytesIO
import pygame
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.getcwd())
from CrabAI.voice.voice_utils import mml_to_audio, audio_to_wave_bytes, calculate_volume_adjustment, create_tone

sr=16000

# Pygameの初期化
pygame.init()
pygame.mixer.init()

for freq in [ 440.0, 220.0, 1, 440+(1000-440)/2, 1000 ]:
    offset = calculate_volume_adjustment( freq, 0.5, 220, 440, 0.6, 1000 )
    print( f"Hz:{freq} offset:{offset}")

MML="t120 cdefgab>cdefgab"
MML="v5 >a"

feed = create_tone( 32, time=0.4, volume=0.01, sample_rate=16000)
while True:
    audio1 = mml_to_audio(MML, sampling_rate=sr)
    audio = np.concatenate( (feed,audio1) )
    wave_bytes = audio_to_wave_bytes( audio, sample_rate=sr )
    wav:BytesIO = BytesIO(wave_bytes)
    wav.seek(0)
    # オンメモリのwaveデータを読み込む
    wave_sound = pygame.mixer.Sound(wav)
    # 再生
    wave_sound.play(fade_ms=0)
    # 音声データの波形をプロット
    #plt.figure(figsize=(10, 4))
    #duration = len(audio)/sr
    #time = np.arange(0, duration, 1/sr).astype(np.float32)  # 時間軸の生成
    #plt.plot(time[:1000], audio[:1000])  # 最初の1000サンプルをプロット
    # plt.plot(time, audio)  # 最初の1000サンプルをプロット
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Waveform of Audio Signal")
    # plt.show()
    MML = input('>>')


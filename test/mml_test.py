import sys,os
import pygame
sys.path.append(os.getcwd())
from CrabAI.voice.voice_utils import mml_to_audio, audio_to_wave_bytes, vol_offset

sr=16000

# Pygameの初期化
pygame.init()
pygame.mixer.init()

for freq in [ 440.0, 220.0, 1, 440+(1000-440)/2, 1000 ]:
    offset = vol_offset( freq, 0.5, 220, 440, 0.6, 1000 )
    print( f"Hz:{freq} offset:{offset}")

MML="t60 cdefgab>cdefgab"
while True:
    audio = mml_to_audio(MML, sampling_rate=sr)
    wav = audio_to_wave_bytes( audio, sample_rate=sr )
    # オンメモリのwaveデータを読み込む
    wave_sound = pygame.mixer.Sound(wav)
    # 再生
    wave_sound.play(fade_ms=0)
    MML = input('>>')


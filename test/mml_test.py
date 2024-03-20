import sys,os
import pygame
sys.path.append(os.getcwd())
from CrabAI.voice.voice_utils import mml_to_sound, audio_to_wave_bytes

MML="cdefgO1cdef"
sr=16000
audio = mml_to_sound(MML,sampling_rate=sr)

# Pygameの初期化
pygame.init()
pygame.mixer.init()

wav = audio_to_wave_bytes( audio,sample_rate=sr )
# オンメモリのwaveデータを読み込む
wave_sound = pygame.mixer.Sound(wav)

while True:
    # 再生
    wave_sound.play(fade_ms=0)
    aa = input('>>')

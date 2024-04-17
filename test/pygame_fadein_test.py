import numpy as np
import pygame
import wave
from pygame.locals import *
from io import BytesIO

def generate_tone(frequency, t1, t2, sample_rate=44100):
    wave1 = np.zeros( int(sample_rate*t1) )
    t = np.linspace(0, t2, int(sample_rate * t2), False)  # 時間軸の作成
    wave2 = np.sin(frequency * t * 2 * np.pi)  # 正弦波の生成
    return np.concatenate( (wave1,wave2) )

def to_wave(audio,sample_rate):

    # 音データを-32768から32767の整数に変換
    pcm = np.int16(audio * 32767).astype(np.int16)

    # BytesIOオブジェクトを作成
    buffer = BytesIO()

    # waveモジュールを使用してwavファイルとして書き込む
    with wave.open(buffer, 'w') as wf:
        wf.setnchannels(1)  # モノラル
        wf.setsampwidth(2)  # サンプルのバイト数（16ビットなので2）
        wf.setframerate(sample_rate)  # サンプリングレート
        wf.writeframes(pcm.tobytes())  # 音声データをバイト形式で書き込み

    # バッファの開始位置に戻る
    buffer.seek(0)

    return buffer

def play_sound(sound_data, sample_rate=44100):
    """音声データを再生する"""
    buffer = BytesIO()
    pygame.mixer.music.load(buffer, 'wav')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)

def main(n1, n2, t1):

    # パラメータ
    n1 = 0.1
    n2 = 0.2
    n3 = 1.0

    pygame.init()
    # サンプリングレート設定
    sample_rate = 44100
    # ミキサー初期化
    pygame.mixer.init(frequency=sample_rate)

    # 440Hzの音生成
    tone_440 = generate_tone(440, n1,n2, sample_rate)
    # 880Hzの音生成
    tone_880 = generate_tone(880, 0,n3, sample_rate)

    # 440Hzの音声を再生
    w1 = to_wave( tone_440, sample_rate )
    w2 = to_wave( tone_880, sample_rate )
    pygame.mixer.music.load(w1)
    pygame.mixer.music.play()
    pygame.mixer.music.queue(w2)

    pygame.time.delay(5000)
    pygame.quit()

# パラメータ
n1 = 1  # 440Hzの音の長さ(秒)
n2 = 1  # 880Hzの音の長さ(秒)
t1 = 1  # 再生の間隔(秒)

# プログラム実行
main(n1, n2, t1)

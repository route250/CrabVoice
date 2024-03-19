import sys,os,io,wave
import numpy as np

#-----------memo

# 一般的な2チャンネル（ステレオ）音声データの形式において、データは通常、左チャンネルと右チャンネルのサンプルが交互に配置される形で格納されます。
# NumPy配列で表す場合、形状は(サンプル数, チャンネル数)となります。
# つまり、2チャンネル音声データでは形状が(n, 2)の配列となり、nはサンプル数（各チャンネルの音声データ点の数）、2は左チャンネルと右チャンネルの2チャンネルを意味します。
#
# 2チャンネル音声データの例（サンプル数3）
#audio_data = np.array([
#    [1, 2],  # 左チャンネルの最初のサンプルと右チャンネルの最初のサンプル
#    [3, 4],  # 左チャンネルの2番目のサンプルと右チャンネルの2番目のサンプル
#    [5, 6]   # 左チャンネルの3番目のサンプルと右チャンネルの3番目のサンプル
#])
#
# audio_data.shape ==> (3,2)

def audio_to_i16( audio ):
    assert isinstance(audio,np.ndarray) and audio.ndim<=2, "invalid datatype"
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        y = audio * 32768
    elif audio.dtype == np.int16:
        y = audio
    else:
        assert False, "invalid datatype"
    return y.astype(np.int16)

def audio_to_pcm16( audio ):
    return audio_to_i16(audio).tobytes()

def audio_to_wave( out, audio, *, samplerate ):
    """np.float32からwaveフォーマットバイナリに変換する"""
    audio_bytes = audio_to_pcm16(audio)
    # wavファイルを作成してバイナリ形式で保存する
    channels = audio.shape[1] if audio.ndim>1 else 1
    with wave.open( out, "wb") as wav_file:
        wav_file.setnchannels(channels)  # ステレオ (左右チャンネル)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(samplerate)  # サンプリングレート
        wav_file.writeframes(audio_bytes)

def audio_to_wave_bytes( audio_f32, *, sample_rate ):
    """np.float32からwaveフォーマットバイナリに変換する"""
    wav_io = io.BytesIO()
    audio_to_wave( wav_io, audio_f32, samplerate=sample_rate )
    wav_io.seek(0)  # バッファの先頭にシーク
    wave_bytes = wav_io.read()
    return wave_bytes

note_to_freq = {
    'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6,
    'E': -5, 'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1,
    'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1, 'B': 2
}

def note_to_hz(note):
    if isinstance(note,int|float):
        return int(note)
    """音名（例:C4）を周波数（Hz）に変換"""
    if note in ['R', 'r', '']:  # 休符の場合
        return 0
    name, octave = note[:-1], int(note[-1])
    return 440.0 * (2 ** ((octave - 4) + note_to_freq[name] / 12.0))

# C4(ド) 261.63, D4(レ) 293.66  E4(ミ) 329.63 F4(ファ) 349.23 G4(ソ) 392.00 A4(ラ) 440.00 B4(シ) 493.88 C5(ド) 523.25
def create_tone(Hz=440, time=0.3, volume=0.3, sample_rate=16000, fade_in_time=0.05, fade_out_time=0.1):
    Hz = note_to_hz(Hz)
    data_len = int(sample_rate * time)
    if Hz > 0:
        # 正弦波を生成
        sound = np.sin(2 * np.pi * np.arange(data_len) * Hz / sample_rate).astype(np.float32)
        # 音量
        sound *= volume
        # フェードイン処理
        fade_in_len = int(sample_rate * fade_in_time)
        fade_in = np.linspace(0, 1, fade_in_len)  # 0から1まで線形に増加
        sound[:fade_in_len] *= fade_in
        # フェードアウト処理
        fade_out_len = int(sample_rate * fade_out_time)
        fade_out = np.linspace(1, 0, fade_out_len)  # 1から0まで線形に減少
        sound[-fade_out_len:] *= fade_out
    else:
        # 無音
        sound = np.zeros(data_len, dtype=np.float32)
    
    return sound

def create_sound(sequence, *, volume:float=0.3 ):
    """複数の（周波数、時間）タプルを受け取り、連続する音声データを生成する

    Args:
        sequence (list of tuples): (Hz, time)のタプルのリスト

    Returns:
        bytes: 生成された音声データのバイナリ（WAV形式）
    """
    sample_rate = 16000
    sounds = [create_tone(Hz, time, volume, sample_rate) for Hz, time in sequence]
    combined_sound = np.concatenate(sounds)
    return audio_to_wave_bytes(combined_sound, sample_rate=sample_rate)


# a1 = np.array( [0,1,2,3], dtype=np.float32 )
# print( f"{a1.shape}")
# a2 = np.array( [[0],[1],[2],[3]], dtype=np.float32 )
# print( f"{a2.shape}")

def fft_audio_signal( audio:np.ndarray, sampling_rate:int ):
    """
    音声データに対してFFTを実行し、周波数スペクトルを計算する。

    Parameters:
    - audio_signal: 音声データを含むnp.float32型のNumPy配列

    Returns:
    - freqs: 周波数成分（Hz）
    - abs_fft: 周波数成分に対応する振幅の絶対値
    """
    # FFTを実行
    fft_result = np.fft.fft(audio)
    # FFT結果の絶対値を取得（複素数から振幅へ）
    abs_fft = np.abs(fft_result)
    
    # 周波数のビンを計算
    freqs = np.fft.fftfreq(len(audio), d=1.0/sampling_rate)
    
    # 振幅スペクトルを半分にして、負の周波数成分を除外
    abs_fft = abs_fft[:len(abs_fft)//2]
    freqs = freqs[:len(freqs)//2]
    
    return freqs, abs_fft

def _voice_rate( freqs, abs_fft, low:float=100.0, high:float=1000.0 ):
    sum_total = 0
    sum_voice = 0
    for freq, amp in zip( freqs, abs_fft ):
        sum_total += amp
        if low<=freq and freq < high:
            sum_voice += amp
    return round( sum_voice/sum_total, 3 )

def voice_per_audio_rate( audio:np.ndarray, sampling_rate:int ):
    freqs, abs_fft = fft_audio_signal(audio, sampling_rate )
    rate = _voice_rate(freqs,abs_fft)
    return rate
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
#    audio_bytes = audio_to_i16(audio)
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

note_to_freq_map = {
    'C-': -10,
    'C': -9,
    'C#': -8,'C+': -8, 'D-': -8,
    'D': -7,
    'D#': -6, 'D+': -6, 'E-': -6,
    'E': -5,
    'F': -4,
    'F#': -3, 'F+': -3, 'G-': -3,
    'G': -2,
    'G#': -1, 'G+': -1,  'A-': -1,
    'A': 0,
    'A#': 1, 'A+': 1, 'B-': 1,
    'B': 2,
    'B#': 3, 'B+': 3,
}

# C4(ド) 261.63, D4(レ) 293.66  E4(ミ) 329.63 F4(ファ) 349.23 G4(ソ) 392.00 A4(ラ) 440.00 B4(シ) 493.88 C5(ド) 523.25
def note_to_freq(note,octave) ->int:
    """音名（例:C4）を周波数（Hz）に変換"""
    base_freq = note_to_freq_map.get(note)
    if base_freq is None: # 休符の場合
        return 0
    return int(440.0 * (2 ** ((octave - 4) + base_freq / 12.0)) )

def create_tone(Hz=440, time=0.3, volume=0.3, sample_rate=16000, fade_in_time=0.01, fade_out_time=0.1):
    data_len = int(sample_rate * time)
    if Hz > 0:
        # 正弦波を生成
        t = np.arange(data_len) / sample_rate
        sound = np.sin(2 * np.pi * Hz * t ).astype(np.float32)
        # ウィンドウ関数の適用（オプション）
        # window = np.hanning(data_len)
        # sound *= window
        # フェードインウィンドウの生成
        fade_in_len = int(sample_rate * min( fade_in_time, time*0.02 ) )
        fade_in_window = np.linspace(0, 1, fade_in_len)
        sound[:fade_in_len] *= fade_in_window
        # フェードアウト処理
        fade_out_len = int(sample_rate *time*0.3 )
        fade_out = np.linspace(1, 0, fade_out_len)  # 1から0まで線形に減少
        sound[-fade_out_len:] *= fade_out
        # 音量
        if volume<1.0:
            sound *= volume
    else:
        # 無音
        sound = np.zeros(data_len, dtype=np.float32)
    
    return sound

def calculate_duration_sec(value, tempo) ->float:
    # 四分音符の基本時間（秒）を計算
    quarter_note_duration = 60 / tempo
    # valueがNoneの場合はデフォルトの音符の長さ（例: 四分音符）
    if value is None:
        value = 4
    return float(quarter_note_duration * (4 / value))  # 音符の長さに応じた時間を計算

def vol_offset(freq, lo, fr1, fr2, hi, fr3):
    if freq < fr1:
        dx = lo / fr1
        return dx * (fr1 - freq)
    elif freq >= fr3:
        return hi
    elif fr2 < freq < fr3:
        dx = hi / (fr3 - fr2)
        return dx * (freq - fr2)
    else:  # これは fr1 <= freq <= fr2 の範囲をカバーします
        return 0.0  # この範囲での補正値が必要な場合は、ここを調整します

def mml_to_audio( mml, *, sampling_rate:int=16000 ):
    sound_list:list = []
    pos:int =0
    mml_len:int = len(mml)
    octave:int = 4  # 初期オクターブ
    tempo:int = 120  # デフォルトテンポ
    default_length:int = 4 # 音符長
    current_vol:int = 7 # デフォルトの音量
    max_vol: int = 15 # 音量最大値
    while pos<mml_len:
        cmd = mml[pos].upper()
        pos += 1
        if cmd==' ':
            continue

        if pos<mml_len and mml[pos] in '#+-':
            cmd += mml[pos]
            pos += 1

        # 数値の読み取り
        val = None
        start_pos = pos
        while pos < mml_len and '0' <= mml[pos] <= '9':
            pos += 1
        if start_pos != pos:
            val = int(mml[start_pos:pos])

        if cmd == 'O' and val is not None:
            octave = val
        elif cmd == '>':
            octave += 1
        elif cmd == '<':
            octave -= 1
        elif cmd == 'T' and val is not None:
            tempo = val
        elif cmd == 'L' and val is not None:
            default_length = val
        elif cmd == 'V' and val is not None:
            current_vol = current_vol = max(0, min(val, max_vol))
        elif cmd in 'CDEFGABR':
            freq = note_to_freq(cmd, octave) if 'A' <= cmd <= 'G' else 0
            ll = val if val is not None else default_length
            duration_sec = calculate_duration_sec( ll, tempo )
            vol = round( current_vol/max_vol, 3 )
            vol = vol + vol_offset( freq, 1.0, 220, 440, 1.0, 1000 )
            print( f"T{tempo} O{octave} V{current_vol} {cmd}{ll} => Hz:{freq} Sec:{duration_sec} Vol:{vol}")
            sound_list.append( create_tone( freq, duration_sec, vol, sampling_rate ) )
        else:
            raise Exception(f"Invalid command: {cmd}")
    return np.concatenate(sound_list,0)

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
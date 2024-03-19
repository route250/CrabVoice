import wave
import numpy as np

class WaveIterator:
    def __init__(self, file_path, chunk_duration=0.1):
        """
        WAVファイルから音声データを読み込み、指定された時間ごとにチャンクを返すイテレータ。

        Parameters:
        - file_path: WAVファイルのパス
        - chunk_duration: チャンクの時間（秒）
        - sampling_rate: サンプリングレート（Hz）
        """
        self.wave_file_path = file_path
        self.chunk_duration = chunk_duration
        self.wave_stream = None
        self.sampling_rate:int = None
        self.frames_per_chunk:int = None
        self.pos:int = 0
        self.start:float = 0
        self.end:float = 0

    def __enter__(self):
        self.wave_stream = wave.open(self.wave_file_path, 'rb')
        self.sampling_rate = int(self.wave_stream.getframerate())
        self.frames_per_chunk = int(self.sampling_rate * self.chunk_duration)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.wave_stream is not None:
            self.wave_stream.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.wave_stream is None:
            raise StopIteration

        frames = self.wave_stream.readframes(self.frames_per_chunk)
        if not frames:
            raise StopIteration
        # WAVファイルのデータをnp.float32形式で読み込み
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        data /= np.iinfo(np.int16).max  # 正規化
        self.start = self.pos/self.sampling_rate
        self.pos+=len(data)
        self.end = self.pos/self.sampling_rate
        return data

class FftVad:

    def __init__(self, *, sampling_rate:int ):
        self.sampling_rate = sampling_rate

    def fft_audio_signal(self, audio_signal ):
        """
        音声データに対してFFTを実行し、周波数スペクトルを計算する。

        Parameters:
        - audio_signal: 音声データを含むnp.float32型のNumPy配列

        Returns:
        - freqs: 周波数成分（Hz）
        - abs_fft: 周波数成分に対応する振幅の絶対値
        """
        # FFTを実行
        fft_result = np.fft.fft(audio_signal)
        # FFT結果の絶対値を取得（複素数から振幅へ）
        abs_fft = np.abs(fft_result)
        
        # 周波数のビンを計算
        freqs = np.fft.fftfreq(len(audio_signal), d=1.0/self.sampling_rate)
        
        # 振幅スペクトルを半分にして、負の周波数成分を除外
        abs_fft = abs_fft[:len(abs_fft)//2]
        freqs = freqs[:len(freqs)//2]
        
        return freqs, abs_fft

    def voice_rate( self, freqs, abs_fft ):
        sum_total = 0
        sum_voice = 0
        for freq, amp in zip( freqs, abs_fft ):
            sum_total += amp
            if 100.0<=freq and freq <= 1000.0:
                sum_voice += amp
        return round( sum_voice/sum_total, 3 )

    def audio_voice_rate( self, audio_signal ):
        freqs, abs_fft = self.fft_audio_signal(audio_signal)
        rate = self.voice_rate(freqs,abs_fft)
        return rate

# 使用例
file_path = 'testData/voice_mosimosi.wav'  # WAVファイルのパスを指定
file_path = 'testData/voice_command.wav'  # WAVファイルのパスを指定

with WaveIterator(file_path, chunk_duration=0.2 ) as audio_chunks:
    sampling_rate = audio_chunks.sampling_rate
    print( f"audio_chunks {sampling_rate}")
    fft:FftVad = FftVad( sampling_rate=sampling_rate )
    for chunk in audio_chunks:
        rate = fft.audio_voice_rate(chunk)
        mark = "*" if rate>0.4 else " "
        start = audio_chunks.start
        end = audio_chunks.end
        print(f"{mark} {start:6.2f}(sec) {rate:6.3f}")


import time
import numpy as np

import librosa
import scipy

class Test:
    def __init__(self):
        self.sampling_rate = 16000
        # 人の声のフィルタリング（バンドパスフィルタ）
        fs_nyq = self.sampling_rate*0.5
        low = 50 / fs_nyq
        high = 1500 /fs_nyq
        self.pass_ba = scipy.signal.butter( 2, [low, high], 'bandpass', output='ba')
        self.cut_ba = scipy.signal.butter( 4, [low, high], 'bandstop', output='ba')

    def generate_gaussian_noise(self, duration_sec, mean=0, std=0.1):
        """
        指定秒数のガウスノイズを生成する。

        Parameters:
        - duration_sec: 生成するノイズの秒数
        - sampling_rate: サンプリングレート（1秒あたりのサンプル数）
        - mean: ガウスノイズの平均値
        - std: ガウスノイズの標準偏差

        Returns:
        - ガウスノイズが含まれるnp.float32形式のNumPy配列
        """
        num_samples = int(duration_sec * self.sampling_rate)
        noise = np.random.normal(mean, std, num_samples).astype(np.float32)
        return noise

    def fft_audio_signal(self, audio_signal ):
        """
        音声データに対してFFTを実行し、周波数スペクトルを計算する。

        Parameters:
        - audio_signal: 音声データを含むnp.float32型のNumPy配列
        - sampling_rate: サンプリングレート（1秒あたりのサンプル数）

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

    def filter(self,audio:np.ndarray):
        xaudio:np.ndarray = audio
        # habs = np.abs(xaudio)
        # nrm = xaudio/habs
        # e1 = librosa.feature.rms( y=xaudio, hop_length=len(nrm))[0][0]
        #s1 = np.sum(habs)
        # # 人の声のフィルタリング（バンドパスフィルタ）
        pass_b,pass_a = self.pass_ba
        #array2 = scipy.signal.sosfilt(self.sos, xaudio)
        pass_audio = scipy.signal.lfilter(pass_b,pass_a,xaudio)
        cut_b,cut_a = self.cut_ba
        cut_audio = scipy.signal.lfilter(cut_b,cut_a,xaudio)
        pass_e = librosa.feature.rms( y=pass_audio, hop_length=len(pass_audio))[0][0]
        cut_e = librosa.feature.rms( y=cut_audio, hop_length=len(cut_audio))[0][0]
        # s2 = np.sum(np.abs(filterd))
        total_e = pass_e + cut_e
        er = cut_e/pass_e
        # sr = s2/s1
        print( f" energy {er} {cut_e}/{pass_e}" )

    def cut(self, audio:np.ndarray):
        cut_b,cut_a = self.cut_ba
        cut_audio = scipy.signal.lfilter(cut_b,cut_a,audio)
        return cut_audio

    def aggregate_fft_to_frequency_bands(self, freqs, abs_fft, band_width=100):
        """
        FFT結果を指定された帯域幅で集計する。

        Parameters:
        - freqs: 周波数成分（Hz）
        - abs_fft: 周波数成分に対応する振幅の絶対値
        - band_width: 集計する帯域の幅（Hz）

        Returns:
        - 集計結果を含む辞書。キーは帯域の中心周波数、値はその帯域における振幅の合計。
        """
        aggregated = {}
        for freq, amp in zip(freqs, abs_fft):
            if freq < 0:
                # 負の周波数は無視
                continue
            band_center = ((freq + (band_width / 2)) // band_width) * band_width
            if band_center not in aggregated:
                aggregated[band_center] = 0
            aggregated[band_center] += amp
        return aggregated

    def simple_text_plot( self, freqs, abs_fft, num_chars=50 ):
        """
        周波数スペクトルをテキストベースでプロットする。

        Parameters:
        - freqs: 周波数成分（Hz）
        - abs_fft: 周波数成分に対応する振幅の絶対値
        - num_chars: 最大振幅を基準とした、1行あたりの最大文字数
        """
        freq_min = min( freqs )
        freq_max = max( freqs )
        max_amp = np.max(abs_fft)
        for freq, amp in zip(freqs, abs_fft):
            bar_chars = int((amp / max_amp) * num_chars)
            bar = '*' * bar_chars
            print(f"{freq:.1f} Hz: {bar}")

    def band_text_plot( self, aggregated:dict, num_chars=50 ):
        """
        周波数スペクトルをテキストベースでプロットする。

        Parameters:
        - freqs: 周波数成分（Hz）
        - abs_fft: 周波数成分に対応する振幅の絶対値
        - num_chars: 最大振幅を基準とした、1行あたりの最大文字数
        """
        max_amp = 0
        for freq, amp in aggregated.items():
            max_amp = amp if amp>max_amp else max_amp
        for freq, amp in aggregated.items():
            bar_chars = int((amp / max_amp) * num_chars)
            bar = '*' * bar_chars
            print(f"{freq:.1f} Hz: {bar}")

def main():
    print( f"----Main----")
    Tst:Test = Test()

    noize:np.ndarray = Tst.generate_gaussian_noise(0.7)
    nrun = 100
    t0 = time.time()
    for n in range(nrun):
        freqs, abs_fft = Tst.fft_audio_signal(noize)
    t0 = time.time()-t0
    te = round( t0/nrun, 4 )
    print( f" FFT {t0}(sec)  {te}(sec/fft)")

    aaa = Tst.aggregate_fft_to_frequency_bands( freqs, abs_fft, band_width=200 )
    Tst.band_text_plot( aaa )
    cut:np.ndarray = Tst.cut( noize )
    cut_freqs, cut_fft = Tst.fft_audio_signal(cut)
    bbb = Tst.aggregate_fft_to_frequency_bands( cut_freqs, cut_fft, band_width=200 )
    Tst.band_text_plot( bbb )

if __name__ == "__main__":
    main()


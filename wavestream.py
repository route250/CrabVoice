import io
import wave
import numpy as np
from threading import Thread,Condition
import time
import pygame

class PipeInpIO:

    def __init__(self):
        self._lock:Condition = Condition()
        self._buf=b''
        self._pos:int = 0
        self._pipe_pos:int = 0
    
    def __enter__(self):
        print(f"__enter__")
        return self

    def seek(self, pos, whence=io.SEEK_SET):
        print(f"seek {pos} {whence}")
        raise io.UnsupportedOperation(f"seek pos:{pos} whence:{whence}")
        # with self._lock:
        #     if whence == io.SEEK_SET:
        #         self._pos = pos
        #     elif whence == io.SEEK_CUR:
        #         self._pos += pos
        #     elif whence == io.SEEK_END:
        #         self._pos = len(self._buf) + pos
        #     # ポインタがバッファの範囲外に移動しないようにする
        #     self._pos = max(0, min(self._pos, len(self._buf)))
    
    def tell(self):
        with self._lock:
            return self._pos

    def read(self, n=None):
        with self._lock:
            if n is None:
                # nが指定されていない場合は、現在の位置から末尾までの全てを読み取る
                b = self._buf[self._pos:]
                self._pos = len(self._buf)
            else:
                # nが指定されている場合は、指定されたバイト数だけ読み取る
                b = self._buf[self._pos:self._pos + n]
                self._pos += len(b)
            return b

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__")

    def _pipe_write(self, b ):
        with self._lock:
            print(f"write")
            self._buf += b
            self._pipe_pos+=len(b)

    def _pipe_flush(self):
        print(f"flush")

    def _pipe_seek(self, pos, whence=io.SEEK_SET):
        print(f"pipe_seek {pos} {whence}")
        with self._lock:
            if whence==io.SEEK_SET:
                next = pos
            elif whence==io.SEEK_CUR:
                next = self._pipe_pos + pos
            else:
                next = -1
            if next<self._pipe_pos:
                raise io.UnsupportedOperation(f"seek pos:{pos} whence:{whence}")
            add = next-self._pipe_pos
            self._buf += b'\0'*add
            self._pipe_pos = next
            return next

    def _pipe_tell(self):
        return self._pipe_pos

class PipeOutIO:

    def __init__(self,inp:PipeInpIO):
        self._inp = inp
    
    def __enter__(self):
        print(f"__enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__")

    def write(self, b):
        self._inp._pipe_write( b )

    def flush(self):
        self._inp._pipe_flush()

    def seek(self, pos, whence=io.SEEK_SET):
        self._inp._pipe_seek( pos, whence )

    def tell(self):
        return self._inp._pipe_tell()

wavfile='nakagawke01.wav'

# # サンプルの音声データを生成する（例：1秒間の440Hzのサイン波）
# sample_rate = 44100  # サンプルレート
# duration = 10  # 秒
# frequency = 440  # Hz
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)  # 16ビットPCMデータ

def rec(wavfile,out):
    bs=800
    with wave.open(wavfile,'rb') as wav_in:
        ch=wav_in.getnchannels()
        ww=wav_in.getsampwidth()
        fr=wav_in.getframerate()
        len=wav_in.getnframes()
        with wave.open(out,'wb') as wav_out:
            wav_out.setnchannels(ch)
            wav_out.setsampwidth(ww)
            wav_out.setframerate(fr)
            wav_out.setnframes(len)
            while True:
                buf = wav_in.readframes(bs)
                if buf is None or len(buf)==0:
                    break
                wav_out.writeframes(buf)

def main():
    # Pygameの初期化
    pygame.init()
    pygame.mixer.init()

    # BytesIOオブジェクトを作成して、waveファイルとして音声データを書き込む
    with PipeInpIO() as inp:
        with PipeOutIO(inp) as out:
            th:Thread = Thread( target=rec, args=(wavfile,out),daemon=True)
            th.start()
            th.join()

        # オンメモリのwaveデータを読み込む
        wave_sound = pygame.mixer.Sound(inp)

        # 再生
        wave_sound.play()

        # 再生が終わるまで待つ
        while pygame.mixer.get_busy():
            time.sleep(1)

    # Pygameの終了処理
    pygame.quit()

if __name__ == "__main__":
    main()
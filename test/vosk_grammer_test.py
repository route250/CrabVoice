
import sys,os,time,json
sys.path.append(os.getcwd())
import wave
import numpy as np
import vosk
from vosk import KaldiRecognizer, Model
vosk.SetLogLevel(-1)
from CrabAI.voice._stt import recognizer_vosk

def load_wave_file( fn:str ) ->np.ndarray:
    with wave.open( fn, 'rb' ) as stream:
        sampling_rate = stream.getframerate()
        nframes:int = stream.getnframes()
        nch:int = stream.getnchannels()
        b = stream.readframes( nframes )
        i16 = np.frombuffer( b, dtype=np.int16 )
        f32 = i16.astype( np.float32 )
        f32 = f32 / 32767.0
        return f32, sampling_rate

def to_pcm_bytes( audio:np.ndarray ) -> bytes:
    return (audio*32767).astype(np.int16).tobytes()

def main():

    input_audio, sample_rate = load_wave_file( 'testData/voice_mosimosi.wav')
    frame_size = int( sample_rate * 2 )
    vosk_model = recognizer_vosk.get_vosk_model(lang="ja")
    gr = recognizer_vosk.get_katakana_grammar()
    #with open( 'grammer.txt', 'w' ) as stream:
    #    stream.write(gr)

    vosk: KaldiRecognizer = KaldiRecognizer(vosk_model, int(sample_rate), gr )

    for s in range( 0, len(input_audio), frame_size ):
        seg:np.ndarray = input_audio[s:s+frame_size]
        pcm = to_pcm_bytes( seg )
        v0 = time.time()
        vosk.AcceptWaveform( pcm )
        txt = vosk.FinalResult()
        vosk.Reset()
        v1 = time.time()
        print( f"{v1-v0}s {txt}" )

if __name__ == "__main__":
    main()
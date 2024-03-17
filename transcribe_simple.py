import sys
import os
import time
import wave
import numpy as np
from faster_whisper import WhisperModel

model_size = "large-v3"

print( f"#Load model")
t0=time.time()
# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
model:WhisperModel = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
t1=time.time()
print( f"# TIME {t1-t0}(sec)")

print( f"#Start")

for sec in range(10,31,10):
    print( f"{sec}(sec)")
    audio30 = np.zeros( 10*16000, dtype=np.float32 )
    t1=time.time()
    segments, info = model.transcribe( audio30, beam_size=5, language='ja')
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    t2=time.time()
    print( f"# TIME {t2-t1}/{sec}(sec)")


print( f"#Start")

t1=time.time()
audio_file='testData/nakagawke01.wav'
segments, info = model.transcribe( audio_file, beam_size=5, language='ja')

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
t2=time.time()
print( f"# TIME {t1-t0} {t2-t1}")

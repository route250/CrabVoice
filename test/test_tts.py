import sys,os
sys.path.append(os.getcwd())
import time

from CrabAI.voice.tts import TtsEngine

aaa = 0
def _tts_callback( text:str, emotion:int, model:str):
    if text:
        aaa = 1
        print( f"[TTS] {text}")
    else:
        aaa = 2
        print( f"[TTS] stop")

def main():
    tts:TtsEngine = TtsEngine( speaker=30, talk_callback=_tts_callback )

    tts.add_talk('あいうえおかきくけこ')
    tts.add_talk('さしすせそたちつてと')
    tts.add_talk('なにぬねのはひふへほ')
    bbb = aaa
    while tts.is_playing():
        if aaa != bbb:
            bbb = aaa
            print(f"{bbb}")
        time.sleep(0.2)

if __name__ == "__main__":
    main()

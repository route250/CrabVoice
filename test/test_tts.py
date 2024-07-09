import sys,os
sys.path.append(os.getcwd())
import time

from CrabAI.voice.tts import TtsEngine

aaa = 0
def _tts_callback( text:str, x, emotion:int, model:str):
    if x==2 or x==3:
        aaa = 1
        print( f"[TTS] {x} {text}")
    elif x==9:
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

# 2024-07-09 22:35:14,174 ERROR [OpenAI] BadRequestError 400 Error code: 400 - {'error': {'message': "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.", 'type': 'invalid_request_error', 'param': 'messages.[2].role', 'code': None}}

#   {\n    "role": "user",\n    "content": "キュン システムって何だ"\n  },\n
#   {\n    "role": "assistant",\n    "content": "{\\"text_to_speech\\": \\"あぁ、お前も気になるのか。よし、調べてみるから待ってろ。\\"}",\n
#     "tool_calls": [\n
#       {\n
#         "id": "call_XeBDke8FSxVhl0kSvEK250cG",\n
#         "function": {\n
#           "name": "google_search_api",\n
#           "arguments": "{\\"keyword\\":\\"きゅんシステム\\"}"\n
#         },\n
#         "type": "function"\n
#       }\n
#     ]\n
#   },\n

#   {\n    "role": "tool",\n    "tool_call_id": "call_XeBDke8FSxVhl0kSvEK250cG",\n    "content": "# Search keyword: きゅんシステム\\n\\n# Search result:\\n\\n1. [撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載 ...](https://otakuma.net/archives/2024070208.html)\\n  7 days ago ... 「なでれば開くシステム」としてSNSに投稿されたのは、寝ている猫を飼い主が撫でている動画。どうやらこの猫ちゃんには、寝ている時に撫でられると指\xa0...\\n\\n2. [撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載 ...](https://topics.smt.docomo.ne.jp/article/otakuma/trend/otakuma-20240702_08)\\n  7 days ago ... 撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載の猫が話題（おたくま経済新聞） 「なでれば開くシステム」としてSNSに投稿されたの\xa0...\\n\\n3. [撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載 ...](https://article.yahoo.co.jp/detail/33461a94f8ab97c353be040df58c5e3d5da5c50c)\\n  7 days ago ... 「なでれば開くシステム」としてSNSに投稿されたのは、寝ている猫を飼い主が撫でている動画。どうやらこの猫ちゃんには、撫でられると指がパカッと\xa0...\\n\\n4. [撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載 ...](https://news.biglobe.ne.jp/trend/0702/9406079230/oks_naeru_hatsudou_neko_01_jpg.html)\\n  写真ニュース(1/1): 撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載の猫が話題. おたくま経済新聞2024年7月2日（火）19時0分. 画像：「きゅん\xa0...\\n\\n5. [るる フワフワプレス社 (@vnsl0o) / X](https://x.com/vnsl0o?lang=ja)\\n  Jun 25, 2024 ... YouTube 見ません #猫 # ... おたくま経済新聞さんに記事化していただきました。 撫でるとパカッと開く手がかわいすぎる！「きゅんシステム」搭載の猫が話題\xa0...\\n\\n"\n  },\n


import json
import re
import os

import traceback
from openai import OpenAI
from openai.types.chat import ChatCompletion

def _convert_to_katakana_openai(word_list, *, openai_timeout=5.0, openai_max_retries=2 ):
    """
    OpenAI APIを使用して英単語リストをカタカナに変換する関数。
    word_list: 変換対象の英単語リスト
    openai_timeout: OpenAI APIのタイムアウト時間
    openai_max_retries: OpenAI APIの最大リトライ回数
    """
    # OpenAIに送るリクエストの作成
    req = '以下の英単語の発音をカタカナに変換して\n\n' + "\n".join(word_list)
    req += "\n\nJSON: { '英単語':'カタカナ',... }"
    request_messages = [
        {'role':'system', 'content':req }
    ]
    # OpenAIのモデルとクライアント設定
    openai_llm_model = 'gpt-3.5-turbo'
    client:OpenAI = OpenAI(timeout=openai_timeout,max_retries=openai_max_retries)
    # OpenAI APIを呼び出し、レスポンスを取得
    res:ChatCompletion = client.chat.completions.create(
            messages=request_messages,
            model=openai_llm_model, max_tokens=4000, temperature=0,
            response_format={"type":"json_object"}
    )
    # APIからのレスポンスをJSON形式で解析
    res_text = res.choices[0].message.content
    res_json = json.loads(res_text)
    #print(res_json)
    return res_json

def _to_prefix(word):
    """
    単語から接頭辞を取得するユーティリティ関数。3文字未満の場合は"ShortWords"を返す。
    """
    return word[:3].replace('/','_') if isinstance(word,str) and len(word)>3 else "ShortWords"

def convert_to_katakana(text, *, cache_dir):
    """
    与えられた文字列内の英単語をカタカナに変換する関数。
    text: 変換対象のテキスト
    """
    # 文字列から英単語を抽出
    unique_words:set[str] = set()
    word=''
    for cc in text+" ":
        if 'a'<=cc<='z' or 'A'<=cc<='Z':
            word+=cc
        elif word and cc in '!#$%&-_/0123456789':
            word+=cc
        elif word:
            unique_words.add(word)
            word=''
    if len(unique_words)==0:
        return text
    
   # 変換後の単語を格納する辞書
    katakana_dict = {}
    # JSONファイルにない単語を格納するリスト
    missing_words = []
    # 必要なJsonファイルの情報を保持する辞書
    json_info_list = {}

    # 必要なJsonファイルをロード
    cache_dir = cache_dir if cache_dir else "."
    for word_prefix in set([_to_prefix(w)for w in unique_words]):
        json_filename = os.path.join(cache_dir,f'{word_prefix}.json')
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as stream:
                content = json.load(stream)
        else:
            content = {}
        json_info_list[word_prefix] = { 'filename': json_filename, 'content': content, 'update':False }
    
    # 英単語をカタカナに変換
    for word in unique_words:
        # 固定変換
        if word == "AI":
            katakana_dict[word] = "エーアイ"
            continue
        if word.lower() == "json":
            katakana_dict[word] = "ジェイソン"
            continue
        json_info:dict = json_info_list[_to_prefix(word)]
        content:dict = json_info.get('content')
        if word in content:
            katakana_dict[word] = content[word]
        else:
            missing_words.append(word)
    
    if missing_words:

        # OpenAI APIを使用してカタカナに変換
        new_katakana = _convert_to_katakana_openai(missing_words)

        # JSONファイルを更新
        if isinstance(new_katakana,dict):
            for word,katakana in new_katakana.items():
                if len(katakana)>0:
                    prefix = _to_prefix(word)
                    json_info:dict = json_info_list.get(prefix)
                    if json_info is None:
                        print(f"[convert_to_katakana] key not found? prefix:{prefix} word:{word}")
                        continue
                    content:dict = json_info.get('content')
                    content[word] = katakana
                    katakana_dict[word] = katakana
                    json_info['update'] = True

        # カタカナ辞書ファイルのディレクトリを作成（存在しない場合）
        os.makedirs(cache_dir,exist_ok=True)

        # 更新があったJSONファイルを保存
        for prefix,json_info in json_info_list.items():
            if json_info.get('update',False):
                with open(json_info['filename'], 'w', encoding='utf-8' ) as stream:
                    json.dump(json_info['content'], stream, ensure_ascii=False, indent=2)
    # 英単語の長さで降順にソート（長い単語から順に処理）
    sorted_words = sorted(katakana_dict.keys(), key=len, reverse=True)
    # 文字列の英単語をカタカナに変換
    for word in sorted_words:
        katakana = katakana_dict[word]
        text = re.sub( re.escape(word), katakana, text, flags=re.IGNORECASE)
    
    # 変換後の文字列を戻り値として返す
    return text

def convert_kuten(text:str):
    text = text.replace("、"," ")
    return text

def main():

    # カタカナ変換辞書ファイルの保存ディレクトリ
    KATAKANA_DICT_DIR='tmp/katakana'

    text = 'this is a pen. これはPythonです。JavaScriptも使います。123は数字です。'
    text = '最近の人気のプログラミング言語はPythonやJavaScriptなどがあるよ！'
    text = 'まあ、最近はね、AIはどんどん進化していますよ'
    while True:
        katakana = convert_to_katakana(text,cache_dir=KATAKANA_DICT_DIR)
        print(katakana)
        kuten = convert_kuten(katakana)
        print(kuten)
        text = input( '>> ' )

if __name__ == "__main__":
     main()
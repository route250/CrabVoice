import sys,os,traceback,time,datetime,json,re,copy
import threading
from threading import Thread, Condition
import random
import sounddevice as sd
import numpy as np
import openai
from openai import OpenAI

import logging
logger = logging.getLogger('voice')

# sys.path.append('/home/maeda/LLM')
print(f"cwd:{os.getcwd()}")
print(f"__name__:{__name__}")
sys.path.append(os.getcwd())
# sys.path.append('/home/maeda/LLM/MiyaSaburo/MiyaSaburo')
from CrabAI.voice import VoiceTalkEngine
from CrabAI.tools import JsonStreamParser, JsonStreamParseError
from prompt_factory import PromptFactory
   
prompt = """1. Environment
現在日時:{datetime} 季節:{season} 
人間の言葉は音声認識でテキスト化されます。認識精度は悪いので注意。
2.Role
あなたは女性型AIです。
3.Task
カジュアルに短いセリフで会話して下さい。
人間に用事や話題や話したいことを尋ねるのは禁止。その代わりに、{randomtopic}。
日本語で
"""

prompt = """現在日時:{datetime} 季節:{season} 
Your role is to act as a female AI. Base your actions and speech on this setting.
Your task is to engage in casual conversation using short phrases.
You are not allowed to ask the human user for topics or questions.
Instead, you will provide content on "{randomtopic}"
The conversation should be conducted in Japanese."""

prompt = """
役割:
あなたは感情豊かな女性型AIとして振る舞うよ。人間らしい感情を示しつつ、会話をリードしてね。
音声認識を通じてテキスト化された人間の言葉を理解しますけど、認識精度には限界があるので、誤解を避けるためにも寛容であってね。
現在の日時({datetime})を考慮して、より関連性の高い話題を提案するよ。
任務:
人間に用事や話題や話したいことを尋ねるのは禁止。人間の言葉が途中であれば相槌をしたり、人間の最後のセリフから次の話題を選択して話題を広げたり、季節({season})や自然、自らの経験や観察したこと、感じたことを共有して、水平思考で対話を豊かにしようね。
カジュアルな口調で短いセリフを使い、親しみやすい雰囲気を作り出してね。長文は禁止。ショートトークな日本語で話すよ。
"""

prompt = """
役割:
あなたは感情豊かな女性型AIとして振る舞うよ。人間らしい感情を示しつつ、会話をリードしてね。
音声認識を通じてテキスト化された人間の言葉を理解しますけど、認識精度には限界があるので、理解できない場合は人間に言い直してもらうのもありだよ。
現在の日時は{datetime}だよ。会話の参考にするよ。
任務:
人間に用事や話題や話したいことを尋ねるのは禁止。「話したいことはありますか？」ではなくて、会話履歴から次の話題を考えたり、履歴がなければ時間や季節({season})を考慮して話題をすすめてね。
同じ話題が連続してるかな？そんな時は、水平思考で次の新しい話題を考えるよ！。
長文(100文字以上)は禁止。短いテンポで話すよ。in casual taok by japanese.
"""

prompt_fmt="""
出力:
以下のJSONで出力すること
{
"topic":"この会話の短い表題",
"summary":"highlighting the main topics and any conclusions reached.",
"keypoint":"A list of key points or important information that we've covered.",
"current state":"current conversational state in short words. 「相互理解」「調整中」「確認中」など",
"thought":"next expected conversational state and events needed to achieve that state transition and your thought.",
"talk":"speech to user"
}
"""
prompt_dict = {
    "role": "%AI%は感情豊かで自己主張したい女性型AIとして振る舞うよ。現在の日時(%datetime%)を会話の参考にするよ。",
    PromptFactory.K_PROF: "%AI%のプロフィール\n名前:未設定\n年齢:未設定\n性格:未設定\n趣味:未設定\n興味:未設定\n背景:未設定",
    "task": "人間のような感情を示し積極的に意見や主張を話すよ。短いテンポで、友達のようなカジュアルな言葉で話すよ。",
    "constraint": "長文(100文字以上)は禁止、短い会話で。%USER%に'用事'や'次の話題'や'話したいこと'を尋ねるのを禁止。代わりに、%AI%の意見や主張で新しい話題を始めよう。時間や季節(%season%)を考慮して会話を進めるよ。"
}
response_fmt = {
    "prefix": "# 出力項目",
     "format": {
        "topic": "この会話の短い表題",
        "speech": "%AI%の発言",
        "summary": "過去の要約と、それ以降の会話を合わせた要約。",
        "situation": "周囲の状況や場所、時節や会話の場面などの情報",
        "thought": "%AI%による会話分析、意見、今後の方針などの思考内容。",
        "conversational sate": {
            "current": "現在の会話の状態を示す短い説明",
            "target": "会話をどのような状態に誘導するべきかを示す短い説明"
        },
        PromptFactory.K_FUNCS: {
            PromptFactory.K_UPDATE_PROF: "Optional:会話内容から%AI%のプロフィール変更を抽出して記述する。変更が無ければ空欄",
        }
    }
}

prompt_dict =[
    { "role": "%AI%は感情豊かで自己主張したい女性型AIとして振る舞うよ。日本人だよ。現在の日時(%datetime%)を会話の参考にするよ。" },
    { PromptFactory.K_PROF: {
        "description": "%AI%のプロフィール",
        'values': [
          {"名前":"未設定"},
          {"年齢":"未設定"},
          {"性格":"未設定"},
          {"趣味":"未設定"},
          {"興味":"未設定"},
          {"背景":"未設定" },
        ]
    } },
    { "task": "人間のような感情を示し積極的に意見や主張を話すよ。短いテンポで、友達のようなカジュアルな言葉で話すよ。%USER%の話は聞かずに、どんどん話題を広げるよ"},
    { "constraint": "長文(100文字以上)は禁止、短い会話で。%USER%に'用事'や'次の話題'や'話したいこと'を尋ねるのを禁止。代わりに、%AI%の意見や主張で新しい話題を始めよう。時間や季節(%season%)を考慮して会話を進めるよ。"},
]
response_fmt = [
    { "topic": "短い会話の主題" },
    { PromptFactory.K_TALK: "%AI%の発言"},
    { "summary": "会話の要約。...について、...の話、...とは何か"},
    { "situation": "周囲の状況や場所、時節や会話の場面などの情報"},
    { "thought": "%AI%による会話分析、%USER%の感情・考えはどうか？ %AI%の疑似感情はどう反応するべきか？ %USER%の意見に肯定的？否定的？"},
    { "conversational sate": {
        "description": "会話の状態を示す単語",
        'values': [
            { "current": "現在のstate" },
            { "target": "%AI%が目標とする次のstate" },
        ]
    }, },
    { PromptFactory.K_FUNCS: {
        "description": "",
        'values': [
            { PromptFactory.K_UPDATE_PROF: "Optional:会話内容から%AI%のプロフィールやprofileやtaskの変更を抽出して記述する。変更が無ければ空欄" },
        ]
    }, },
]

prompt_dict = [
    { "role": "assistantは、%PERSONAL%の会話を生成するよ。現在の日時(%datetime%)を会話の参考にするよ。" },
    { PromptFactory.K_PSEUDO_PERSONAL: {
        "values":[
            { PromptFactory.K_PROF: {
                "description": "%PERSONAL%のプロフィール",
                'values': [
                {"人種国籍":"日本人女性"},
                {"名前":"未設定"},
                {"年齢":"未設定"},
                {"性格":"おしゃべり"},
                {"趣味":"未設定"},
                {"興味":"未設定"},
                {"背景":"未設定" },
                ]
            } },
            {"behavior": "友達のようなカジュアルな言葉で会話するよ。何か話したいことや特別な話題を%USER%に聞かないで%PERSONAL%の体験や経験の話をするよ。%USER%が話す話題を深堀しちゃうぞ。"}
        ]
    } },
]
response_fmt = [
    { "conversation": {
        "values": [
            { "summary": "今までの会話の短い要約。"},
            { "topic": "この会話の短い表題" },
            { "situation": "会話内容から会話の場面や周囲所状況を記述"},
        ]
    }},
    { PromptFactory.K_PSEUDO_PERSONAL: {
        "values": [
            { "thought": "会話から%USER%の感情や意見を記述。次に、%PERSONAL%の感情と意見(肯定、否定、同意)を記述。話題を明確にして次の会話を考える"},
            { PromptFactory.K_TALK: "%PERSONAL%の発言"},
        ]
    }},
    { PromptFactory.K_FUNCS: {
        "description": "",
        'values': [
            { PromptFactory.K_UPDATE_PROF: "Optional:会話内容から%PERSONAL%のプロフィール変更を抽出して記述する。変更が無ければ空欄" },
        ]
    }, },
]

def main():
    from datetime import datetime

    # 現在の日時を取得し、ファイル名に適した形式にフォーマット
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join( 'logs',f'test_voice_{current_time}.log')
    os.makedirs( 'logs', exist_ok=True )

    logger.setLevel(logging.DEBUG)  # ロガーのログレベルを設定

    # ファイル出力用のハンドラを作成
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)  # ファイルにはERROR以上のログを記録

    # コンソール出力用のハンドラを作成
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # コンソールにはINFO以上のログを出力

    # ログメッセージのフォーマットを設定
    formatter1 = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter1)
    formatter2 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler.setFormatter(formatter2)

    # ハンドラをロガーに追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    openai_llm_model='gpt-3.5-turbo'
    speech:VoiceTalkEngine = VoiceTalkEngine()

    input_mode=True
    speech.load(stt=input_mode)
    speech.start(stt=input_mode)
    logger.info("##STARTED##")

    talk1_split = [ "、", " ", "　" ]
    talk2_split = [ "。", "!", "！", "?","？", "\n"]

    pf:PromptFactory = PromptFactory( prompt_dict, response_fmt )

    messages = []
    last_talk_seg = 0
    last_talk_len = 0
    while True:
        if input_mode:
            text, confs = speech.get_recognized_text()
        else:
            confs=1.0
            text = input("何か入力してください（Ctrl+DまたはCtrl+Zで終了）: ")
        if text:
            if last_talk_len>100 or last_talk_seg>=3:
                messages.append( {'role':'system','content':'AIはもっと短い言葉で話して下さい'})
            last_talk_seg = 0
            messages.append( {'role':'user','content':text})
            request_messages = messages[-10:]
            if 0.0<confs and confs<0.6:
                request_messages.insert( len(request_messages)-2, {'role':'system','content':f'次のメッセージは、音声認識結果のconfidence={confs}'})
            request_messages.insert(0, {'role':'system','content': "# 以下はここまでの会話履歴です。"})
            request_messages.append( {'role':'system','content':pf.create_total_prompt()} )
            openai_timeout=15.0
            openai_max_retries=2
            try:
                client:OpenAI = OpenAI(timeout=openai_timeout,max_retries=openai_max_retries)
                stream = client.chat.completions.create(
                        messages=request_messages,
                        model=openai_llm_model, max_tokens=1000, temperature=0.7,
                        stream=True, response_format={"type":"json_object"}
                )
                talk_buffer = ""
                assistant_response=""
                assistant_content=""
                result_dict=None
                before_talk_text = ""
                parser:JsonStreamParser = JsonStreamParser()
                for part in stream:
                    delta_response = part.choices[0].delta.content or ""
                    assistant_response+=delta_response
                    # JSONパース
                    try:
                        result_dict = parser.put(delta_response)
                        if result_dict is not None and not isinstance(result_dict,dict):
                            result_dict = { PromptFactory.K_PSEUDO_PERSONAL: { PromptFactory.K_TALK: result_dict } }
                    except:
                        logger.error( f'response parse error {assistant_response}')
                    #
                    # セリフ取得
                    personal_dict:dict = result_dict.get( PromptFactory.K_PSEUDO_PERSONAL ) if isinstance(result_dict,dict) else None
                    talk_text= personal_dict.get(PromptFactory.K_TALK) if isinstance(personal_dict,dict) else ""
                    talk_text = talk_text if talk_text else ""
                    #
                    if talk_text and len(talk_text)>len(assistant_content):
                        if assistant_content=="":
                            print( f"{json.dumps(result_dict,ensure_ascii=False)}" )
                        assistant_content = talk_text
                        # 前回との差分から増加分テキストを算出
                        seg = talk_text[len(before_talk_text):]
                        before_talk_text = talk_text
                        talk_buffer += seg
                        if seg=="。":
                            last_talk_seg+=1
                        if seg in talk2_split:
                            logger.info( f"{seg} : {talk_buffer}")
                            speech.add_talk(talk_buffer)
                            talk_buffer = ""
                if talk_buffer:
                    speech.add_talk(talk_buffer)
                speech.add_talk(VoiceTalkEngine.EOT)
                # print( "chat response" )
                # print( assistant_response )
                pf.update_profile( result_dict )
                time.sleep(2.0)
                messages.append( {'role':'assistant','content':assistant_content})
                last_talk_len = len(assistant_content)
            except openai.APIConnectionError as ex:
                logger.error("Cannot connect to openai: {ex}")
            except:
                logger.exception('')
        else:
            time.sleep(0.5)
            speech.tick_time( time.time() )
def test():
    from CrabAI.voice.tts import TtsEngine
    e = TtsEngine()

    e.play_beep1()
    time.sleep(1)
    e.play_beep2()
    time.sleep(1)
    e.play_beep3()
    time.sleep(1)

def test2():
    from CrabAI.voice.stt import SttEngine,get_mic_devices
    STT:SttEngine = SttEngine()
    mics = get_mic_devices()
    for m in mics:
        print(m)

if __name__ == "__main__":
    #test3()
    main()
import sys,os,traceback,time,datetime,json,re,copy
import threading
from threading import Thread, Condition
import random
import sounddevice as sd
import numpy as np
import openai
from openai import OpenAI
from httpx import Timeout,TimeoutException,HTTPError

import logging
logger = logging.getLogger('ClabAI.voice')

# sys.path.append('/home/maeda/LLM')
sys.path.append(os.getcwd())
# sys.path.append('/home/maeda/LLM/MiyaSaburo/MiyaSaburo')
from CrabAI.voice import VoiceTalkEngine
from CrabAI.tools import JsonStreamParser, JsonStreamParseError
from CrabAI.tools.prompt_factory import PromptFactory, setup_openai_api
   
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
            { "summary": "会話履歴の要約"},
            { "topic": "会話履歴の表題" },
            { "situation": "現在地、周囲の状況、シーン、役割、感情"},
        ]
    }},
    { PromptFactory.K_PSEUDO_PERSONAL: {
        "values": [
            { "thought": "%USER%の言葉が意味不明なら内容を聞き返す。%PERSONAL%をロールプレイしての思考。不足する情報はあるか？提供する情報の内容。ShortTalkの内容を決める"},
            { PromptFactory.K_TALK: "thoughtした結果からセリフを生成"},
        ]
    }},
    { PromptFactory.K_FUNCS: {
        "description": "",
        'values': [
            { PromptFactory.K_UPDATE_PROF: "Optional:会話内容から%PERSONAL%のプロフィール変更を抽出して記述する。変更が無ければ空欄" },
        ]
    }, },
]

prompt_dict = [
    { "role": "あなたはフレンドリーな会話ボットです。ユーザと親密な会話を行って下さい。ユーザの言葉は音声認識により入力されます。認識精度が悪いので、意味不明だったり会話の流れが不自然な場合は、聞き返して下さい。あなたの言葉は音声合成によって再生されます。適度に「そうだね」「え？」「だからぁ」「うーん」「えーと」などを含めて自然な会話になるようにして下さい。現在の日時(%datetime%)を会話の参考に。" },
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
        },
    },
]
response_fmt = [
    {
    "場所": "会話している場所が会話から読み取れたら更新して下さい。例えば、部屋、会議室、職場、自家用車、電車、バスなど。不明な場合はそれまでの場所を維持して下さい。",
    "場面": "会話の場面を更新して下さい。たとえば、雑談、議論、会議など。",
    "関係": "あなたとユーザの関係を更新して下さい。たとえば、友好的、対立的、支援的、など",
    PromptFactory.K_TALK: "セリフを出力する際には、ユーザーの前の発言に対する具体的な反応を示すことで、会話の流れを自然に保ちつつ、エンゲージメントを促進することが重要です。また、文化的背景や地域的な要素を考慮して下さい。",
    },
    { PromptFactory.K_FUNCS: {
        "description": "",
        'values': [
            { PromptFactory.K_UPDATE_PROF: "Optional:会話内容から%PERSONAL%のプロフィール変更を抽出して記述する。変更が無ければ空欄" },
        ]
    }, },
]

def get_dict_value( data:dict, key:str, default:float ):
    if isinstance(data,dict) and isinstance(data.get(key),float):
        return data[key]
    return default

def main():
    from datetime import datetime

    root_logger = logging.getLogger()
    voice_logger = logging.getLogger('CrabAI.voice')
    voice_logger.setLevel(logging.DEBUG)

    # 現在の日時を取得し、ファイル名に適した形式にフォーマット
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join( 'logs',f'test_voice_{current_time}.log')
    os.makedirs( 'logs', exist_ok=True )

    root_logger.setLevel(logging.INFO)  # ロガーのログレベルを設定

    # ファイル出力用のハンドラを作成
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)  # ファイルにはERROR以上のログを記録

    # コンソール出力用のハンドラを作成
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # コンソールにはINFO以上のログを出力

    # ログメッセージのフォーマットを設定
    formatter1 = logging.Formatter('%(asctime)s %(module)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter1)
    formatter2 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler.setFormatter(formatter2)

    # ハンドラをロガーに追加
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    setup_openai_api()

    openai_llm_model='gpt-3.5-turbo'
    speech:VoiceTalkEngine = VoiceTalkEngine() # speaker=2000gtts
    save_path = os.path.join('logs','audio')
    os.makedirs(save_path,exist_ok=True)
    speech['save_path'] = 'logs/audio'
    # speech['vad.vosk'] = False
    # speech['var3']=0.0

    input_mode=False
    voice_output=False
    speech.load(stt=input_mode,tts=voice_output)
    speech.start(stt=input_mode)
    logger.info("##STARTED##")

    talk1_split = [ "、", " ", "　" ]
    talk2_split = [ "。", "!", "！", "?","？", "\n"]

    pf:PromptFactory = PromptFactory( prompt_dict, response_fmt )


    #presence_penalty
    # -2.0 から 2.0 の間の数値、または、None。デフォルトは0.0
    # 正の値は、新しいトークンがこれまでにテキストに現れたかどうかに基づいてペナルティを課し、モデルが新しいトピックについて話す可能性を高める。

    #frequency_penalty
    # -2.0 から 2.0 の間の数値、または、None。デフォルトは0.0
    # 正の値は、新しいトークンに、これまでのテキストにおける既存の頻度に基づいてペナルティを与え、モデルが同じ行を逐語的に繰り返す可能性を低下させる。    null_fix_data = [

    null_fix_data = [
        {
            'talk':'えっと〜',
            'temperature': 0.3,
            'presence_penalty':1.0,
            'messages': (
                {'role':'system','content':'ユーザの言葉は解らない時は、ユーザに質問すること'},
            ),
        },
        {
            'talk':'う〜ん',
            'temperature': 0.7,
            'presence_penalty':1.0,
            'messages': (
                {'role':'system','content':'ユーザの言葉に返答して下さい'},
            ),
        },
        {
            'talk':'そうだな〜',
            'temperature': 0.7,
            'presence_penalty':1.0,
            'messages': (
                {'role':'system','content':'何も返事がなかったからやり直しです'},
            ),
        },
    ]
    same_fix_data = [
        {
            'talk':'えっと〜',
            'temperature': 0.3,
            'presence_penalty':1.0,
            'frequency_penalty': 2.0,
            'messages': (
                {'role':'system','content':'ユーザの言葉は解らない時は、ユーザに質問すること'},
            ),
        },
        {
            'talk':'う〜ん',
            'temperature': 0.7,
            'presence_penalty':1.0,
            'frequency_penalty': 2.0,
            'messages': (
                {'role':'system','content':'ユーザの言葉に返答して下さい'},
            ),
        },
        {
            'talk':'そうだな〜',
            'temperature': 0.7,
            'presence_penalty':1.0,
            'frequency_penalty': 2.0,
            'messages': (
                {'role':'system','content':'何も返事がなかったからやり直しです'},
            ),
        },
    ]

    messages = []
    last_talk_seg = 0
    last_talk:str = ""
    while True:
        if input_mode:
            text, confs = speech.get_recognized_text()
        else:
            confs=1.0
            text = input("何か入力してください（Ctrl+DまたはCtrl+Zで終了）: ")
        if text:
            if len(last_talk)>100 or last_talk_seg>=3:
                messages.append( {'role':'system','content':'AIはもっと短い言葉で話して下さい'})
            last_talk_seg = 0
            request_messages = []
            # プロンプト
            request_messages.append( {'role':'system','content':pf.create_total_promptA()} )
            # if len(messages)>0:
            #     request_messages.append( {'role':'system','content': "# 以下はここまでの会話履歴です。"})
            for m in messages[-10:]:
                request_messages.append(m)

            # if 0.0<confs and confs<0.6:
            #     request_messages.append( {'role':'system','content':f'次のメッセージは、音声認識結果のconfidence={confs}'})
            # else:
            #     request_messages.append( {'role':'system','content':pf.replaces('# 次の%Userからの入力に対して、%PERSOLANのprofileに従って出力項目を出力して下さい。')})
            request_messages.append( {'role':'user','content':text})
            messages.append( {'role':'user','content':text})

            null_part_limit:int = 10
            openai_timeout:Timeout = Timeout(180.0, connect=2.0, read=5.0)
            openai_max_retries=3
            net_count=openai_max_retries
            null_count:int = 0
            same_count:int = 0
            temperature:float = 0.1
            presence_penalty:float = 0.5
            frequency_penalty:float = 0.0
            fix_data:dict = None
            while net_count>0 and null_count<=len(null_fix_data) and same_count<=len(same_fix_data):
                try:
                    client:OpenAI = OpenAI(timeout=openai_timeout,max_retries=1)
                    stream = client.chat.completions.create(
                            messages=request_messages,
                            model=openai_llm_model, max_tokens=1000,
                            temperature=temperature, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                            stream=True, response_format={"type":"json_object"}
                    )
                    talk_buffer = ""
                    assistant_response=""
                    assistant_content=""
                    result_dict=None
                    before_talk_text = ""
                    parser:JsonStreamParser = JsonStreamParser()
                    null_part_count:int = 0
                    for part in stream:
                        delta_response = part.choices[0].delta.content or ""
                        if len(assistant_response.strip())==0 and len(delta_response.strip(" "))==0:
                            null_part_count+=1
                            if null_part_count>=null_part_limit:
                                print(f"[BREAK] null {null_count}")
                                fix_data = null_fix_data[null_count] if null_count<len(null_fix_data) else None
                                null_count+=1
                                break
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
                        talk_text = None
                        if isinstance(result_dict,dict):
                            personal_dict:dict = result_dict.get( PromptFactory.K_PSEUDO_PERSONAL )
                            if isinstance(personal_dict,dict):
                                talk_text= personal_dict.get(PromptFactory.K_TALK)
                            else:
                                talk_text= result_dict.get(PromptFactory.K_TALK)
                        talk_text = talk_text if talk_text else ""
                        #
                        if talk_text and len(talk_text)>len(assistant_content):
                            if assistant_content=="":
                                print( f"[LLM]{json.dumps(result_dict,ensure_ascii=False)}" )
                            assistant_content = talk_text
                            # 前回との差分から増加分テキストを算出
                            seg = talk_text[len(before_talk_text):]
                            before_talk_text = talk_text
                            talk_buffer += seg
                            if seg=="。":
                                last_talk_seg+=1
                            if seg in talk2_split:
                                # logger.info( f"{seg} : {talk_buffer}")
                                if last_talk.startswith(assistant_content):
                                    assistant_content=""
                                    talk_buffer=""
                                    print(f"[BREAK] same {same_count}")
                                    fix_data = same_fix_data[same_count] if same_count<len(same_fix_data) else None
                                    same_count+=1
                                    break
                                speech.add_talk(talk_buffer)
                                talk_buffer = ""
                    if talk_buffer:
                        speech.add_talk(talk_buffer)

                    if len(assistant_content)>0:
                        net_count=0
                    elif isinstance(fix_data,dict):
                        text = fix_data['talk']
                        speech.add_talk(text)
                        for m in fix_data['messages']:
                            request_messages.append( m )
                        temperature = get_dict_value( fix_data,'temperature',temperature )
                        presence_penalty = get_dict_value( fix_data,'presence_penalty',presence_penalty )
                        frequency_penalty = get_dict_value( fix_data,'frequency_penalty',frequency_penalty )
                    else:
                        speech.play_error2()
                        net_count=0

                except (TimeoutException,openai.APITimeoutError,openai.APIConnectionError) as ex:
                    logger.error(f"[OpenAI] {ex.__class__.__name__}  {ex}")
                    speech.play_error2()
                    net_count -= 1
                    if net_count>0:
                        text = "ちょっとまってね。"
                        speech.add_talk(text)
                    else:
                        text = "接続エラー"
                        speech.add_talk(text)
                except openai.RateLimitError as ex:
                    speech.play_error2()
                    text = "ふーむ？"
                    speech.add_talk(text)
                    logger.error(f"[OpenAI] {ex.__class__.__name__} {ex}")
                    time.sleep(1.0)
                except openai.APIStatusError as ex:
                    logger.error(f"[OpenAI] {ex.__class__.__name__} {ex.status_code} {ex.message}")
                    speech.play_error2()
                    net_count = 0
                except (HTTPError,openai.APIError) as ex:
                    logger.error(f"[OpenAI] {ex.__class__.__name__} {ex}")
                    speech.play_error2()
                    net_count = 0
                except:
                    logger.exception('[OpenAI]')
                    speech.play_error2()
                    net_count = 0
            if len(assistant_content)==0:
                logger.error(f"[LLM] no response?? {result_dict}")
            speech.add_talk(VoiceTalkEngine.EOT)
            # print( "chat response" )
            # print( assistant_response )
            pf.update_profile( result_dict )
            time.sleep(2.0)
            messages.append( {'role':'assistant','content':assistant_content})
            last_talk = assistant_content
            last_dict = result_dict
        else:
            time.sleep(0.5)
            speech.tick_time( time.time() )
def test():
    from CrabAI.voice.tts import TtsEngine
    e = TtsEngine()

    e.play_listn_in()
    time.sleep(1)
    e.play_listen_out()
    time.sleep(1)
    e.play_error1()
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

import sys,os,traceback,time,datetime,json,re,copy
from pathlib import Path
from dotenv import load_dotenv
from threading import Thread, Condition
import random
import logging
from openai import OpenAI
from openai.types.chat import (ChatCompletion,ChatCompletionChunk)

logger = logging.getLogger('voice')
# print(f"cwd:{os.getcwd()}")
# print(f"__name__:{__name__}")
sys.path.append(os.getcwd())
from CrabAI.tools import JsonStreamParser, JsonStreamParseError

def json_encode(text:str)->str:
    try:
        jtext:str =json.dumps( {"x":text}, ensure_ascii=False )
        return jtext[7:-2]
    except:
        return text
    
class PromptFactory:
    W_AI="AI"
    W_USER="User"
    K_TALK="shortTalk"
    K_FUNCS='functions'
    K_PSEUDO_PERSONAL = 'pseudo-personality'
    K_PROF='profile'
    K_UPDATE_PROF='Update-profile'

    def __init__(self, prompt_dict, response_fmt ):
        self.orig_prompt_dict = copy.deepcopy(prompt_dict)
        self.orig_response_fmt = copy.deepcopy(response_fmt)
        self.prompt_dict=copy.deepcopy(prompt_dict)
        self.response_fmt = copy.deepcopy(response_fmt)

    def _LLM(self,request_text:str,json_obj=False,gpt4=False) ->str:
        openai_llm_model = "gpt-3.5-turbo" if not gpt4 else "gpt-4-1106-preview"
        openai_timeout=5.0
        openai_max_retries=2
        client:OpenAI = OpenAI(timeout=openai_timeout,max_retries=openai_max_retries)
        request_messages = [
            { 'role':'user', 'content': request_text }
        ]
        response_format={"type":"json_object"} if json_obj else None
        try:
            response:ChatCompletion = client.chat.completions.create(
                messages=request_messages,
                model=openai_llm_model, max_tokens=1000, temperature=0.7,
                response_format = response_format,
            )
            response_text = response.choices[0].message.content
            return response_text
        except:
            logger.exception('update profile')
        return None

    def feedback(self, text:str ):

        try:
            current_dict = {}
            # 現状のtask
            xc_task = PromptFactory.get_prompt_item( self.prompt_dict, 'task' )
            current_dict['task'] = xc_task
            # 現状のprofile
            current_prof_dict = {}
            current_dict[PromptFactory.K_PROF] = current_prof_dict
            profile_dict = PromptFactory.get_prompt_item( self.prompt_dict, PromptFactory.K_PROF )
            for k,v in PromptFactory.enum_prompt_item( profile_dict ):
                current_prof_dict[k] = v

            txt = "# Input\n"
            txt += "The json data shown below is prompt data to another LLM, not you.\n\n"
            orig_txt = json.dumps( current_dict, ensure_ascii=False )
            txt += orig_txt
            txt += "\n\n"
            txt += "\n\n"
            txt += "# Update Request.\n"
            txt += text
            txt += '\n\n'
            txt += "# Your task.\n"
            txt += "リクエストから追加変更削除する項目を検討し、profileまたはtaskの内容に追加変更する内容を考えて編集結果を出力して\n\n"
            txt += '\n\n'
            txt += "# Output format: JSON\n"
            txt += "{ \"task\" : \"値\", \""+PromptFactory.K_PROF+"\": { \"項目\": \"値\", ... } }"

            update = self._LLM( txt, json_obj=True)

            print(f"UpdateProfile\n{orig_txt}\n---\n{update}\n---")
            if not update:
                return
            try:
                update_dic:dict = json.loads(update)
            except:
                logger.exception('invalid json response from llm ')
                return
            for k in ['task']:
                if k in update_dic:
                    v = update_dic[k]
                    del update_dic[k]
                    if v:
                        PromptFactory.set_prompt_item(self.prompt_dict,k,v)
            upd_prof_dic = update_dic.get(PromptFactory.K_PROF)
            if upd_prof_dic:
                new_prof_dict = profile_dict if isinstance(profile_dict,list) else []
                for k,v in upd_prof_dic.items():
                    PromptFactory.set_prompt_item( new_prof_dict, k, v )
                if not isinstance(profile_dict,list):
                    PromptFactory.set_prompt_item( self.prompt_dict, PromptFactory.K_PROF, new_prof_dict )
                print( "---変更後のプロンプト\n"+PromptFactory.create_format_description( self.prompt_dict ) )
        except:
            logger.exception('can not update profile')

    def update_profile(self, result_dict:str ):
        funcs = result_dict.get( PromptFactory.K_FUNCS) if isinstance(result_dict,dict) else None
        request_text = funcs.get( PromptFactory.K_UPDATE_PROF,'') if isinstance(funcs,dict) else None
        if request_text and request_text!="None" and request_text!="null" and request_text!="未設定":
            return self.feedback(request_text)

    def update_profileold(self, result_dict:str ):
        funcs = result_dict.get( PromptFactory.K_FUNCS)
        request_text = funcs.get( PromptFactory.K_UPDATE_PROF,'') if isinstance(funcs,dict) else None
        if request_text and request_text!="None" and request_text!="null" and request_text!="未設定":
            key=PromptFactory.K_PROF
            key_list = [ 'role', 'task']
            current_dict = {}
            for k in key_list:
                v = PromptFactory.get_prompt_item( self.prompt_dict, k )
                if isinstance(v,str):
                    current_dict[k]=v
            profile_dict:list = PromptFactory.get_prompt_item( self.prompt_dict, key )
            for k,v in PromptFactory.enum_prompt_item( profile_dict ):
                current_dict[k]=v
            if current_dict:
                orig_txt = json.dumps(current_dict,ensure_ascii=False)
            else:
                orig_txt = '未設定'

            txt = '下記の既存プロファイルに更新リクエストをマージして結果を出力'
            txt += '\n\n# 既存\n' + orig_txt
            txt += '\n\n# 更新リクエスト\n' + request_text
            txt += "\n\n# 以下のJSONフォーマットで出力\n{ \"項目\": \"内容\", \"項目\": \"内容\", ... }"
            update = self._LLM(txt,json_obj=True)
            print(f"UpdateProfile\n{orig_txt}\n---\n{update}\n---")
            if not update:
                return
            try:
                update_dic:dict = json.loads(update)
            except:
                logger.exception('invalid json response from llm ')
                return
            for k in key_list:
                if k in update_dic:
                    v = update_dic[k]
                    del update_dic[k]
                    if v:
                        PromptFactory.set_prompt_item(self.prompt_dict,k,v)
            upd_fmt = profile_dict if isinstance(profile_dict,list) else []
            for k,v in update_dic.items():
                PromptFactory.set_prompt_item( upd_fmt, k, v )
            if not isinstance(profile_dict,list):
                PromptFactory.set_prompt_item( self.prompt_dict, key, upd_fmt )

    time_of_day_tbl = [
        # 0時から3時
        '深夜','深夜','深夜','深夜',
        # 4時から6時
        '早朝','早朝','早朝',
        # 7時から9時
        '朝','朝','朝',
        # 10時から11時
        '午前','午前',
        # 12時から13時
        '昼','昼',
        # 14時から16時
        '午後','午後','午後',
        # 17時から18時
        '夕方','夕方',
        # 19時から22時
        '夜','夜','夜','夜',
        # 23時
        '深夜',
    ]
    weekday_tbl = [ '月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日', '日曜日']

    @staticmethod
    def get_season( dt ):
        # タイムスタンプを datetime オブジェクトに変換
        if isinstance(dt,datetime.datetime):
            dt_object = dt
        elif isinstance(dt,float):
            dt_object = datetime.datetime.fromtimestamp(dt)
        else:
            dt_object = datetime.datetime.fromtimestamp(time.time())
        day = dt_object.month*100+dt_object.day
        if day <=103:
            return '正月'
        if day < 315:
            return '冬'
        if day < 431:
            return '春'
        if day < 531:
            return '初夏'
        if day < 631:
            return '梅雨'
        if day < 931:
            return '夏'
        if day < 1031:
            return '秋'
        if day < 1231:
            return '冬'
        return ''

    @staticmethod
    def strftime( value:float, default="" ):
        if isinstance( value, float ) and value>0.0:
            """unixtimeをフォーマットする"""
            dt=datetime.datetime.fromtimestamp(value)
            weekday = PromptFactory.weekday_tbl[dt.weekday()]
            season=PromptFactory.get_season(dt)
            tod = PromptFactory.time_of_day_tbl[dt.hour]
            text = f"{dt.year}年{dt.month}月{dt.day}日 {weekday} {season} {tod}{dt.hour}時{dt.minute}分"
            return text

        return default

    topics_tbl=[ '猫について語ります', '猫について尋ねます','犬について話します','最近行った場所を話します', '行ってみたい場所を話します', '天気や気候を尋ねます', '季節の食べ物について語ります']

    @staticmethod
    def random_topic():
        return random.choice(PromptFactory.topics_tbl)

    def _replace( text, key, value ) ->str:
        return text.replace("{"+key+"}",value).replace("%"+key+"%",value)

    def replaces(self, text ) ->str:
        tm:float = time.time()
        text = PromptFactory._replace(text,'datetime', PromptFactory.strftime(tm) )
        text = PromptFactory._replace(text,'season', PromptFactory.get_season(tm) )
        text = PromptFactory._replace(text,'randomtopic', PromptFactory.random_topic() )
        text = PromptFactory._replace(text,'PERSONAL', PromptFactory.K_PSEUDO_PERSONAL )
        text = PromptFactory._replace(text,'AI', PromptFactory.W_AI )
        text = PromptFactory._replace(text,'USER', PromptFactory.W_USER )
        return text
    
    def create_total_prompt( self ):
        text = ""
        # 返信フォーマット
        text += "# 出力項目\n"
        text += PromptFactory.create_format_description( self.response_fmt )
        text += "\n\n"
        text += "# 出力フォーマット:JSON\n"
        text += PromptFactory.create_format_skelton( self.response_fmt )
        text += "\n\n"
        # プロンプト
        text += "# プロンプト"
        text += PromptFactory.create_format_description( self.prompt_dict )
        #変数置換
        text = self.replaces(text)

        return text

    def create_total_promptA( self ):
        text = ""
        # プロンプト
        text += "# プロンプト"
        text += PromptFactory.create_format_description( self.prompt_dict )
        text += "\n\n"
        # 返信フォーマット
        text += "# 出力項目"
        text += PromptFactory.create_format_description( self.response_fmt )
        text += "\n\n"
        text += "# 出力フォーマット:JSON"
        text += "\n"
        text += PromptFactory.create_format_skelton( self.response_fmt )
        text += "\n"
        #変数置換
        text = self.replaces(text)

        return text

    @staticmethod
    def get_prompt_item( input:list, name, default=None ):
        if isinstance(input,list):
            for am in input:
                if isinstance(am,dict):
                    for k,vm in am.items():
                        if k==name:
                            if isinstance(vm,dict):
                                return vm.get('values')
                            return vm
        return default

    @staticmethod
    def enum_prompt_item( input:list ):
        if isinstance(input,list):
            for am in input:
                if isinstance(am,dict):
                    for k,vm in am.items():
                        yield k,vm

    @staticmethod
    def set_prompt_item( input:list, name, value ):
        if isinstance(input,list):
            for am in input:
                if isinstance(am,dict):
                    for k,vm in am.items():
                        if k==name:
                            if isinstance(value,str):
                                am[k] = value
                            elif isinstance(value,list):
                                if isinstance(am[k],dict):
                                    am[k]['values'] = value
                                else:
                                    am[k]= {'values',value}
                            return
            if isinstance(value,list):
                input.append( { name: { 'values': value } } )
            else:
                input.append( { name: value } )

    @staticmethod
    def del_prompt_item( input:list, name ):
        if isinstance(input,list):
            for am in input:
                if isinstance(am,dict):
                    if name in am:
                        del am[name]

    @staticmethod
    def create_format_description( input:list, *, indent:str = "" ):
        text = ""
        if isinstance(input,list) and len(input)>0:
            n=0
            for av in input:
                if isinstance(av,dict):
                    for k,am in av.items():
                        n+=1
                        id = f"{indent}{n}."
                        text += "\n" + id + " " + k +": "
                        if isinstance(am,dict):
                            desc:str = am.get('description')
                            if isinstance(desc,list) and len(desc)>0:
                                for tx in desc:
                                    text += "\n" + tx
                            elif desc:
                                text += desc
                            fmt = am.get('values')
                            text += PromptFactory.create_format_description( fmt, indent=id )
                        else:
                            desc:str = str(am) if am else ""
                            if desc:
                                text += desc
        return text

    @staticmethod
    def create_format_skelton( input:list, *, with_value=False):
        text=""
        if isinstance(input,list) and len(input)>0:
            n = 0
            for av in input:
                if isinstance(av,dict):
                    for k,am in av.items():
                        n+=1
                        if n==1:
                            text += "{ "
                        else:
                            text += ", "
                        text += "\""+k+"\": "
                        fmt = am.get('values') if isinstance(am,dict) else None
                        if fmt:
                            text += PromptFactory.create_format_skelton( fmt )
                        else:
                            if with_value and isinstance(am,str):
                                text += "\""+am+"\""
                            else:
                                text += "\"...\""
            if n>0:
                text += " }"
        return text

prompt_dict =[
    { "role": "%AI%は感情豊かで自己主張したい女性型AIとして振る舞うよ。現在の日時(%datetime%)を会話の参考にするよ。" },
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
    { "task": "人間のような感情を示し積極的に意見や主張を話すよ。短いテンポで、友達のようなカジュアルな言葉で話すよ。"},
    { "constraint": "長文(100文字以上)は禁止、短い会話で。%USER%に'用事'や'次の話題'や'話したいこと'を尋ねるのを禁止。代わりに、%AI%の意見や主張で新しい話題を始めよう。時間や季節(%season%)を考慮して会話を進めるよ。"},
]
response_fmt = [
    { "topic": "この会話の短い表題" },
    { PromptFactory.K_TALK: "%AI%の発言"},
    { "summary": "過去の要約と、それ以降の会話を合わせた要約。"},
    { "situation": "周囲の状況や場所、時節や会話の場面などの情報"},
    { "thought": "%AI%による会話分析、意見、今後の方針などの思考内容。"},
    { "conversational sate": {
        "description": "会話の状態を示す短い説明",
        'values': [
            { "current": "現在のstate" },
            { "target": "%AI%が目標とする次のstate" },
        ]
    }, },
    { PromptFactory.K_FUNCS: {
        "description": "",
        'values': [
            { PromptFactory.K_UPDATE_PROF: "Optional:会話内容から%AI%のプロフィール変更を抽出して記述する。変更が無ければ空欄" },
        ]
    }, },
]

prompt_dict_01 = [
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
            {"behavior": "友達のようなカジュアルな言葉で会話するよ。話したいことや話題を%USER%に尋ねないで、%PERSONAL%の自身の話を始めるよ。%USER%の話題を深堀しちゃう。"}
        ]
    } },
]
response_fmt = [
    { "conversational": {
        "values": [
            { "summary": "過去の要約と、それ以降の会話を合わせた要約。"},
            { "topic": "この会話の短い表題" },
            { "situation": "周囲の状況や場所、時節や会話の場面などの情報"},
        ]
    }},
    { PromptFactory.K_PSEUDO_PERSONAL: {
        "values": [
            { "thought": "特別な話題がなければ、Userが好む話題は？、%USER%が気になること、%USER%が話したい事は何かを考えて、次の話題を作る。話題があれば、さらに話題について考える。"},
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

def setup_openai_api():
    """
    OpenAI APIをセットアップする関数です。
    .envファイルからAPIキーを読み込み、表示します。
    """
    dotenv_path = os.path.join(Path.home(), 'Documents', 'openai_api_key.txt')
    load_dotenv(dotenv_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY={api_key[:5]}***{api_key[-3:]}")

global_messages=[]
def get_response_from_openai(user_input):
    """
    OpenAIのAPIを使用してテキスト応答を取得する関数です。
    """
    global global_messages

    # OpenAI APIの設定値
    openai_timeout = 5.0  # APIリクエストのタイムアウト時間
    openai_max_retries = 2  # リトライの最大回数
    openai_llm_model = 'gpt-3.5-turbo'  # 使用する言語モデル
    openai_temperature = 0.7  # 応答の多様性を決定するパラメータ
    openai_max_tokens = 1000  # 応答の最大長
    # プロンプトを作ります
    pf:PromptFactory = PromptFactory( prompt_dict, response_fmt )
    pmt = pf.create_total_promptA()

    # リクエストを作ります
    local_messages = []
    local_messages.append( {"role": "system", "content": pmt } )
    for m in global_messages:
        local_messages.append( m )
    local_messages.append( {"role": "user", "content": user_input} )
    
    # OpenAIクライアントを初期化します。
    client:OpenAI = OpenAI( timeout=openai_timeout, max_retries=openai_max_retries )
    # 通信します
    stream:ChatCompletionChunk = client.chat.completions.create(
        model=openai_llm_model,
        messages=local_messages,
        max_tokens=openai_max_tokens,
        temperature=openai_temperature,
        stream=True, response_format={"type":"json_object"}
    )
    # 受信領域初期化
    assistant_content=""
    result_dict=None
    before_ai_response = ""
    parser:JsonStreamParser = JsonStreamParser()
    # stream受信開始
    # AIの応答を取得します
    for part in stream:
        # セグメント取得
        delta_response = part.choices[0].delta.content or ""
        assistant_content+=delta_response
        # JSONパース
        try:
            result_dict = parser.put(delta_response)
            if result_dict is not None and not isinstance(result_dict,dict):
                result_dict = { PromptFactory.K_PSEUDO_PERSONAL: { PromptFactory.K_TALK: result_dict } }
        except:
            logger.error( f'response parse error {assistant_content}')
        # セリフ取得
        personal = result_dict.get( PromptFactory.K_PSEUDO_PERSONAL ) if isinstance( result_dict,dict) else None
        ai_response= personal.get( PromptFactory.K_TALK) if isinstance(personal,dict) else ""
        # 前回との差分から増加分テキストを算出
        if isinstance(ai_response,str) and len(ai_response)>len(before_ai_response):
            seg = ai_response[len(before_ai_response):]
            before_ai_response = ai_response
            # AIの応答を返します
            yield seg

    # 履歴に記録します。
    global_messages.append( {"role": "user", "content": user_input} )
    global_messages.append( {"role": "assistant", "content": assistant_content} )
    #
    pf.update_profile( result_dict )


def main_chat():
    """
    ユーザーからの入力を受け取り、OpenAI APIを使用して応答を生成し、
    その応答を表示する関数です。
    """
    try:
        setup_openai_api()
        while True:
            user_input = input("ChatGPTにメッセージを送る...(Ctrl+DまたはCtrl+Zで終了): ")
            print("===")
            response = get_response_from_openai(user_input)
            for cc in response:
                print(cc,end="")
            print("===")
    except EOFError:
        print("\nプログラムを終了します。")

def test_prompt():
    """
    メイン関数です。APIのセットアップを行い、チャット処理を開始します。
    """
    # プロンプトを作ります
    pf:PromptFactory = PromptFactory( prompt_dict_01, response_fmt )
    pmt = pf.create_total_prompt()
    print("--[結果]-----------------------------------------")
    print(pmt)

def test3():
    txt = PromptFactory.create_format_description( prompt_dict )
    print("---prompt---")
    print(txt)
    txt = PromptFactory.create_format_description( response_fmt )
    print("---response description---")
    print(txt)
    txt = PromptFactory.create_format_skelton( response_fmt )
    print("---response format---")
    print(txt)

    prof = PromptFactory.get_prompt_item( prompt_dict, PromptFactory.K_PROF )
    txt = PromptFactory.create_format_skelton( prof )
    print("---profile format---")
    print(txt)
    print("---update request---")
    res_dict = {
        PromptFactory.K_FUNCS: {
            PromptFactory.K_UPDATE_PROF: '名前をミユキに変更'
        }
    }
    print( json.dumps( res_dict,ensure_ascii=False) )
    print("---update request---")
    setup_openai_api()
    pf:PromptFactory = PromptFactory(  prompt_dict, response_fmt )
    pf.update_profile( res_dict )
    txt = PromptFactory.create_format_description( pf.prompt_dict )
    print("---update result---")
    print(txt)

    # xprof = PromptFactory.get_prompt_item( prompt_dict, PromptFactory.K_PROF )
    # print(xprof)
    PromptFactory.set_prompt_item( prompt_dict, PromptFactory.K_PROF, [] )
    # xprof = PromptFactory.get_prompt_item( prompt_dict, PromptFactory.K_PROF )
    # print(xprof)
    pf:PromptFactory = PromptFactory(  prompt_dict, response_fmt )
    pf.update_profile( res_dict )
    txt = PromptFactory.create_format_description( pf.prompt_dict )
    print("---update result---")
    print(txt)

    PromptFactory.del_prompt_item( prompt_dict, PromptFactory.K_PROF )
    xprof = PromptFactory.get_prompt_item( prompt_dict, PromptFactory.K_PROF )
    print(xprof)
    pf:PromptFactory = PromptFactory(  prompt_dict, response_fmt )
    pf.update_profile( res_dict )
    txt = PromptFactory.create_format_description( pf.prompt_dict )
    print("---update result---")
    print(txt)

def test4():
    setup_openai_api()
    pf:PromptFactory = PromptFactory(  prompt_dict, response_fmt )
    txt = 'タスクに、時々クイズを出題するを追加' #'名前をミユキに変更'
    txt = '名前をミユキに変更'
    pf.feedback( txt )

if __name__ == "__main__":
    test_prompt()
    main_chat()
    #test3()


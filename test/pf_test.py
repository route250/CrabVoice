import sys,os,traceback,time,datetime,json,re,copy
import random
import numpy as np


sys.path.append(os.getcwd())
from prompt_factory import PromptFactory, setup_openai_api

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

def main():

    pf:PromptFactory = PromptFactory( prompt_dict, response_fmt )

    print("[PROMPT]------------------------")
    print(pf.create_total_promptA())
    print("--------------------------------")

if __name__ == "__main__":
    main()
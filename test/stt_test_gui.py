
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import pygame

import sys,os
sys.path.append(os.getcwd())
from CrabAI.voice._stt.audio_to_text import AudioToText, SttData
from CrabAI.voice.voice_utils import audio_to_wave_bytes
from stt_data_plot import SttDataPlotter

# 音声解析関数
def analysis_audio(wavefilename, callback):
    STT:AudioToText = AudioToText( callback=callback )
    STT.load( filename=wavefilename )
    STT.start()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('音声解析GUI')
        self.geometry('800x600')
        self.results = {}
        self.create_widgets()

    def create_widgets(self):

        # ボタンを配置するフレーム
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(fill=tk.X, pady=10)

        # ファイル選択ボタン
        self.load_button = tk.Button(self.button_frame, text='ファイルを選択', command=self.load_file)
        self.load_button.pack(side=tk.LEFT)

        # 実行ボタン
        self.run_button = tk.Button(self.button_frame, text='実行', command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT)

        # 再生ボタン
        self.play_button = tk.Button(self.button_frame, text='再生(audio)', command=self.play_audio)
        self.play_button.pack(side=tk.LEFT)

        # 再生ボタン
        self.play_button = tk.Button(self.button_frame, text='再生(raw)', command=self.play_raw)
        self.play_button.pack(side=tk.LEFT)

        # 停止ボタン
        self.stop_button = tk.Button(self.button_frame, text='停止', command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT)

        # 結果表示テーブル
        self.tree = ttk.Treeview(self, columns=('utc','start', 'end', 'sec', 'content'), show='headings')
        self.tree.heading('utc', text='utc')
        self.tree.heading('start', text='開始フレーム')
        self.tree.heading('end', text='終了フレーム')
        self.tree.heading('sec', text='長さ')
        self.tree.heading('content', text='結果テキスト')
        # カラム幅の設定
        self.tree.column('utc', width=30)
        self.tree.column('start', width=30)
        self.tree.column('end', width=30)
        self.tree.column('sec', width=30)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 音声波形グラフ
        self.plot1 = SttDataPlotter(self)
        self.plot1.pack(fill=tk.BOTH, expand=True)

        # 選択イベントのバインド
        self.tree.bind('<<TreeviewSelect>>', self.on_item_select)

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.filename:
            print(f"ファイルが選択されました: {self.filename}")

    def run_analysis(self):
        if hasattr(self, 'filename'):
            self.plot(None)
            self.results={}
            for item in self.tree.get_children():
                self.tree.delete(item)
            analysis_audio(self.filename, self.update_result)
        else:
            print("ファイルが選択されていません")

    def update_result(self, stt_data:SttData):
        if stt_data.typ != SttData.Text and stt_data.typ != SttData.Dump:
            return
        row_id = len(self.results)  # 現在の行数をキーとする
        self.results[row_id] = stt_data  # 結果を辞書に追加
        # 結果をテーブルに表示
        utc = stt_data.utc
        sec = (stt_data.end-stt_data.start)/stt_data.sample_rate
        self.tree.insert('', tk.END, values=(utc, stt_data.start, stt_data.end, sec, stt_data.content))
        self.plot(stt_data)

    def on_item_select(self, event):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        stt_data = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        self.plot(stt_data)

    def plot(self,stt_data:SttData):
        self.plot1.plot(stt_data)

    def play_audio(self):
        self.play_audiox(False)

    def play_raw(self):
        self.play_audiox(True)

    def play_audiox(self,b):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        stt_data:SttData = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        if stt_data is None:
            print("stt_data is None")
            return
        st_sec, ed_sec = self.plot1.get_xlim()
        if b:
            audio = stt_data.raw
        else:
            audio = stt_data.audio
        if audio is None:
            print("stt_data is None")
            return

        st = max(0, int(st_sec * stt_data.sample_rate) - stt_data.start)
        ed = min( len(audio), int(ed_sec * stt_data.sample_rate) - stt_data.start )
        bb = audio_to_wave_bytes( audio[st:ed], sample_rate=stt_data.sample_rate)
        xx = BytesIO(bb)
        pygame.mixer.init()
        pygame.mixer.music.load(xx)
        pygame.mixer.music.play()

    def stop_audio(self):
        pygame.mixer.music.stop()

if __name__ == '__main__':
    app = Application()
    app.mainloop()

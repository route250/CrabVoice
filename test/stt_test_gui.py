
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
        self.play_button = tk.Button(self.button_frame, text='再生', command=self.play_audio)
        self.play_button.pack(side=tk.LEFT)

        # 停止ボタン
        self.stop_button = tk.Button(self.button_frame, text='停止', command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT)

        # 結果表示テーブル
        self.tree = ttk.Treeview(self, columns=('start', 'end', 'sec', 'content'), show='headings')
        self.tree.heading('start', text='開始フレーム')
        self.tree.heading('end', text='終了フレーム')
        self.tree.heading('sec', text='長さ')
        self.tree.heading('content', text='結果テキスト')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 音声波形グラフ
        self.figure = plt.Figure(figsize=(6,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        if stt_data.typ != SttData.Text:
            return
        row_id = len(self.results)  # 現在の行数をキーとする
        self.results[row_id] = stt_data  # 結果を辞書に追加
        # 結果をテーブルに表示
        sec = (stt_data.end-stt_data.start)/stt_data.sample_rate
        self.tree.insert('', tk.END, values=(stt_data.start, stt_data.end, sec, stt_data.content))
        self.plot(stt_data)

    def on_item_select(self, event):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        stt_data = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        self.plot(stt_data)

    def plot(self,stt_data:SttData):
        self.figure.clear()
        if stt_data is not None and stt_data.hists is not None:
            # 音声波形を描画
            hists:pd.DataFrame = stt_data.hists
            hi:np.ndarray = hists['hi']
            lo:np.ndarray = hists['lo']
            co:np.ndarray = hists['color']
            vad:np.ndarray = hists['vad1']

            frames = stt_data.end - stt_data.start
            chunks = len(hi)
            chunk_size = frames//chunks

            # 上下2つのサブプロットを作成
            ax1 = self.figure.add_subplot(2, 1, 1)  # 上のサブプロット
            ax2 = self.figure.add_subplot(2, 1, 2)  # 下のサブプロット
            ax3 = ax2.twinx()
            sec = [ f for f in range(stt_data.start, stt_data.end )]
            x_sec = [ round((stt_data.start + (i*chunk_size))/stt_data.sample_rate,3) for i in range(len(hi)) ]
            x_positions = np.arange(len(hi))

            #ax1.plot( x_sec, hi,  color='gray' )
            #ax1.plot( x_sec, lo, color='gray' )
            ax1.fill_between( x_sec, lo, hi, color='gray')
            ax1.set_ylim( ymin=-1.0, ymax=1.0)
            ax1.grid(True)

            # 上のサブプロットのX軸ラベルと目盛りを非表示にする
            #ax1.set_xticklabels([])
            #ax1.set_xticks([])  # 目盛り自体を非表示にする
            #ax1.set_xlabel('')  # X軸ラベルを非表示にする

            ax2.plot( x_sec, vad, color='r', label='vad' )
            ax2.set_ylim( ymin=0.0, ymax=1.5 )
            ax2.grid(True)
            ax3.step( x_sec, co, where='post', color='b', label='color' )
            ax3.set_ylim( ymin=0.0, ymax=3.0 )

            h2, l2 = ax2.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            ax2.legend( h2+h3, l2+l3, loc='upper right')

            self.canvas.draw()

    def play_audio(self):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        stt_data:SttData = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        if stt_data is not None and stt_data.audio is not None:
            bb = audio_to_wave_bytes( stt_data.audio, sample_rate=stt_data.sample_rate)
            xx = BytesIO(bb)
            pygame.mixer.init()
            pygame.mixer.music.load(xx)
            pygame.mixer.music.play()
        else:
            print("ファイルが選択されていません")

    def stop_audio(self):
        pygame.mixer.music.stop()

if __name__ == '__main__':
    app = Application()
    app.mainloop()

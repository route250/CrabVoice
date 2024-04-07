
from io import BytesIO
import traceback
from threading import Thread
from queue import Queue, Empty
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
from stt_data_plot import SttDataTable, SttDataPlotter

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('音声解析GUI')
        self.geometry('800x600')
        self.create_widgets()

        self._ev_queue = Queue()
        self.running = True
        self.after_id=None
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # ウィンドウが閉じられたときのイベントハンドラを設定
        self._idle_loop()

        self.stt=None

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
        self.table = SttDataTable(self)
        self.table.bind( self.on_item_select )
        self.table.pack(fill=tk.BOTH, expand=True)

        # 音声波形グラフ
        self.plot1 = SttDataPlotter(self)
        self.plot1.pack(fill=tk.BOTH, expand=True)

    def on_close(self):
        self.running = False  # runningフラグをFalseに設定してループを停止
        try:
            self.stt.stop()
        except:
            pass
        try:
            self.after_cancel(self.after_id)
        except:
            pass
        self.destroy()  # ウィンドウを閉じる

    def _idle_loop(self):
        if self.running:
            try:
                task = self._ev_queue.get_nowait()
                task()
            except Empty:
                pass
            except:
                traceback.print_exc()
            self.after_id = self.after( 200, self._idle_loop )

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"),("Stt files", "*.npz")])
        if self.filename:
            print(f"ファイルが選択されました: {self.filename}")

    # 音声解析関数
    def analysis_audio(self):
        self.stt:AudioToText = AudioToText( callback=self.update_result )
        self.stt.load( filename=self.filename )
        self.stt.start()

    def run_analysis(self):
        if hasattr(self, 'filename'):
            self.plot(None)
            self.table.clear()
            t = Thread( target=self.analysis_audio, daemon=True )
            t.start()
        else:
            print("ファイルが選択されていません")

    def update_result(self, stt_data:SttData):
        self._ev_queue.put( lambda stt_data=stt_data,file_path=None: self.table.add(stt_data, file_path=file_path) )
        #self.table.add(stt_data)
        # self.plot(stt_data)

    def on_item_select(self, stt_data:SttData ):
        try:
            self.plot(stt_data)
        except:
            traceback.print_exc()

    def plot(self,stt_data:SttData):
        self.plot1.plot(stt_data)

    def play_audio(self):
        self.play_audiox(False)

    def play_raw(self):
        self.play_audiox(True)

    def play_audiox(self,b):
        stt_data:SttData = self.table.selected()
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

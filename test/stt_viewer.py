
import sys,os,glob
import traceback
from threading import Thread
from queue import Queue, Empty
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import pygame

sys.path.append(os.getcwd())
from CrabAI.voice._stt.audio_to_text import SttData
from stt_data_plot import SttDataTable, SttDataPlotter

# SttDataを表示、再生するGUI

class SttDataViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('音声解析GUI')
        self.geometry('800x600')
        self.results = {}
        self.create_widgets()
        self.dir_path = '.'
        self._ev_queue = Queue()
        self.running = True
        self.after_id=None
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # ウィンドウが閉じられたときのイベントハンドラを設定
        self._idle_loop()

    def create_widgets(self):

        # ボタンを配置するフレーム
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, pady=4)

        # ファイル選択ボタン
        self.load_button = ttk.Button(self.button_frame, text='フォルダを選択', command=self.load_from_directory)
        self.load_button.pack(side=tk.LEFT)

        self.main_frame = ttk.PanedWindow( self )
        self.main_frame.pack( fill=tk.BOTH, expand=True )

        # 結果表示テーブル
        self.table = SttDataTable(self.main_frame)
        self.table.bind( self.on_item_select )
        self.table.pack(fill=tk.BOTH, expand=True)

        # 音声波形グラフ
        self.plot1 = SttDataPlotter(self.main_frame)
        self.plot1.pack(fill=tk.BOTH,expand=True)

        self.main_frame.add(self.table)
        self.main_frame.add(self.plot1)

    def on_close(self):
        self.running = False  # runningフラグをFalseに設定してループを停止
        try:
            self.after_cancel(self.after_id)
        except:
            pass
        self.destroy()  # ウィンドウを閉じる
        try:
            pygame.mixer.quit()
        except:
            pass

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

    def load_from_directory(self):
        try:
            self.dir_path = filedialog.askdirectory( initialdir=self.dir_path )
            if os.path.isdir(self.dir_path):
                self.table.clear()
                t = Thread( target=self._th_load_from_directory, args=(self.dir_path,), daemon=True )
                t.start()
        except:
            print(f"ロードできません")

    def _th_load_from_directory(self, dir_path):
        try:
            if os.path.isdir(dir_path):
                self.table.clear()
                pattern = os.path.join( self.dir_path, 'audio*.npz')
                file_path_list = glob.glob(pattern)
                for file_path in file_path_list:
                    try:
                        stt_data:SttData = SttData.load(file_path)
                        if stt_data is not None:
                            if stt_data.typ == SttData.Text or stt_data.typ == SttData.Dump:
                                max_vad = max(stt_data['vad'])
                                if max_vad>0.2:
                                    self._ev_queue.put( lambda stt_data=stt_data,file_path=file_path: self.table.add(stt_data, file_path=file_path) )
                        else:
                            print(f"ロードできません: {file_path}")
                    except:
                        print(f"ロードできません: {file_path}")
        except:
            print(f"ロードできません: {file_path}")

    def on_item_select(self, stt_data:SttData):
        self.plot(stt_data)

    def plot(self,stt_data:SttData):
        # GraphPlotterを使用してグラフを描画
        self.plot1.set_stt_data(stt_data)

if __name__ == '__main__':
    app = SttDataViewer()
    app.mainloop()


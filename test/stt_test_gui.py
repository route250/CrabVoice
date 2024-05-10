
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

def _getvalue(entry):
    try:
        return float(entry.get())
    except ValueError:
        return None

def _setvalue(ent,value):
    ent.delete(0, tk.END)
    ent.insert(0, value )

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
        self.menu_bar = tk.Menu
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, pady=10)

        # ファイル選択ボタン
        self.load_button = ttk.Button(self.button_frame, text='ファイルを選択', command=self.load_file)
        self.load_button.pack(side=tk.LEFT)

        self.fn_text = ttk.Label( self.button_frame, text="..." )
        self.fn_text.pack(side=tk.LEFT, padx=1)

        # fpassの設定
        ttk.Label(self.button_frame, text="fpass:").pack(side=tk.LEFT,ipadx=5)
        self.fpass_entry = ttk.Entry(self.button_frame, width=4)
        self.fpass_entry.pack(side=tk.LEFT, padx=1)
        
        # fstopの設定
        ttk.Label(self.button_frame, text="fstop:").pack(side=tk.LEFT)
        self.fstop_entry = ttk.Entry(self.button_frame,width=4)
        self.fstop_entry.pack(side=tk.LEFT, padx=5)
        
        # gpassの設定
        ttk.Label(self.button_frame, text="gpass:").pack(side=tk.LEFT)
        self.gpass_entry = ttk.Entry(self.button_frame,width=4)
        self.gpass_entry.pack(side=tk.LEFT, padx=5)
        
        # gstopの設定
        ttk.Label(self.button_frame, text="gstop:").pack(side=tk.LEFT)
        self.gstop_entry = ttk.Entry(self.button_frame,width=4)
        self.gstop_entry.pack(side=tk.LEFT, padx=5)

        stt:AudioToText = AudioToText(callback=None)
        fpass, fstop, gpass, gstop = stt['vad.butter']
        self._set_butter( fpass, fstop, gpass, gstop )

        # 実行ボタン
        self.run_button = ttk.Button(self.button_frame, text='実行', command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT)

        # 結果表示テーブル
        self.table = SttDataTable(self)
        self.table.bind( self.on_item_select )
        self.table.pack(fill=tk.BOTH, expand=True)

        # 音声波形グラフ
        self.plot1 = SttDataPlotter(self)
        self.plot1.pack(fill=tk.BOTH, expand=True)

    def _set_butter(self, fpass, fstop, gpass, gstop ):
        _setvalue(self.fpass_entry,fpass)
        _setvalue(self.fstop_entry,fstop)
        _setvalue(self.gpass_entry,gpass)
        _setvalue(self.gstop_entry,gstop)

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

    def _th_load_from_directory(self, file_path:str):
        try:
            if not os.path.isfile(file_path):
                print(f"ロードできません: {file_path}")
                return
            if file_path.endswith('.npz') or file_path.endswith('.wav'):
                stt_data:SttData = SttData.load(file_path)
                if stt_data is None:
                    print(f"ロードできません: {file_path}")
                    return
                if stt_data.typ != SttData.Text and stt_data.typ != SttData.Dump:
                    print(f"ロードできません: {file_path}")
                    return
                if self.stt is not None:
                    self.stt.stop()
                    self.stt = None
                self.filename = file_path
                filename_only = os.path.basename(self.filename)
                self.fn_text.configure(text=filename_only)
                self.plot(None)
                self.table.clear()
                self._ev_queue.put( lambda stt_data=stt_data,file_path=file_path: self.table.add(stt_data, file_path=file_path) )
        except:
            print(f"ロードできません: {file_path}")

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Stt files","*"),("WAV files", "*.wav"),("all files", "*")])
        if filename:
            self._th_load_from_directory(filename)

    # 音声解析関数
    def analysis_audio(self):
        self.stt:AudioToText = AudioToText( callback=self.update_result )

        butter = self.stt['vad.butter']
        for idx, ent in enumerate( [ self.fpass_entry, self.fstop_entry, self.gpass_entry, self.gstop_entry ] ):
            try:
                val = float(ent.get())
                butter[idx] = val
            except ValueError:
                ent.delete(0, tk.END)
                ent.insert(0, butter[idx] )
        self.stt['vad.butter'] = butter

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
        self.plot1.set_stt_data(stt_data)

if __name__ == '__main__':
    app = Application()
    app.mainloop()

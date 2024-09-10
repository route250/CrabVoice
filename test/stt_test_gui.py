
from typing import Literal
from io import BytesIO
import traceback
import time
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
from CrabAI.vmp import ShareParam
from CrabAI.voice.stt import SttEngine, SttData
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

        self.stt_engine:SttEngine|None=None
        self._cwd:str = os.getcwd()
        self._current_file:SttData|None = None
        self._current_stt_data:SttData|None = None

    def create_widgets(self):

        self.menu_bar = tk.Menu

        # ファイル表示フレーム
        self.file_frame = ttk.Frame(self)
        self.file_frame.pack(fill=tk.BOTH, pady=1,expand=True)

        # ファイル表示フレームのボタンを配置するフレーム
        self.file_bar = ttk.Frame(self.file_frame)
        self.file_bar.pack(fill=tk.X, pady=10,expand=True)

        # ファイル選択ボタン
        self.load_file_button = ttk.Button(self.file_bar, text='file', command=self.load_file)
        self.load_file_button.pack(side=tk.LEFT)
        self.load_dir_dbutton = ttk.Button(self.file_bar, text='dir', command=self.load_dir)
        self.load_dir_dbutton.pack(side=tk.LEFT)

        self.seg_open_button = ttk.Button(self.file_bar, text='▼', command=lambda:self._select_table(True))
        self.seg_open_button.pack(side=tk.RIGHT)
        self.file_table = SttDataTable(self.file_frame)
        self.file_table.bind( self.on_file_select )
        self.file_table.pack(fill=tk.BOTH, expand=True)

        # 結果表示フレームのボタンを配置するフレーム
        self.seg_bar = ttk.Frame(self.file_frame)
        self.seg_bar.pack(fill=tk.X, pady=10,expand=True)

        self.fn_text = ttk.Label( self.seg_bar, text="..." )
        self.fn_text.pack(side=tk.LEFT, padx=1)

        # fpassの設定
        ttk.Label(self.seg_bar, text="fpass:").pack(side=tk.LEFT,ipadx=5)
        self.fpass_entry = ttk.Entry(self.seg_bar, width=4)
        self.fpass_entry.pack(side=tk.LEFT, padx=1)
        
        # fstopの設定
        ttk.Label(self.seg_bar, text="fstop:").pack(side=tk.LEFT)
        self.fstop_entry = ttk.Entry(self.seg_bar,width=4)
        self.fstop_entry.pack(side=tk.LEFT, padx=5)
        
        # gpassの設定
        ttk.Label(self.seg_bar, text="gpass:").pack(side=tk.LEFT)
        self.gpass_entry = ttk.Entry(self.seg_bar,width=4)
        self.gpass_entry.pack(side=tk.LEFT, padx=5)
        
        # gstopの設定
        ttk.Label(self.seg_bar, text="gstop:").pack(side=tk.LEFT)
        self.gstop_entry = ttk.Entry(self.seg_bar,width=4)
        self.gstop_entry.pack(side=tk.LEFT, padx=5)

        conf:ShareParam = ShareParam()
        SttEngine.load_default(conf)
        fpass, fstop, gpass, gstop = conf.get_audio_butter()
        self._set_butter( fpass, fstop, gpass, gstop )

        # 実行ボタン
        self.run_button = ttk.Button(self.seg_bar, text='実行', command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT)

        self.seg_close_button = ttk.Button(self.seg_bar, text='▲', command=lambda:self._select_table(False))
        self.seg_close_button.pack(side=tk.RIGHT)

        self.seg_table = SttDataTable(self.file_frame)
        self.seg_table.bind( self.on_seg_select )
        self.seg_table.pack(fill=tk.BOTH, expand=True)

        self._select_table(False)

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
            if self.stt_engine:
                self.stt_engine.stop()
        except:
            pass
        try:
            if self.after_id:
                self.after_cancel(self.after_id)
        except:
            pass
        self.destroy()  # ウィンドウを閉じる

    def _select_table(self,b:bool):
        if b:
            if self._current_file:
                self.file_bar.pack_forget()
                self.file_table.pack_forget()
                if self._current_stt_data is None or self._current_stt_data != self._current_file:
                    self._current_stt_data = self._current_file
                    self.seg_table.clear()
                    self.seg_table.add(self._current_stt_data, self._current_stt_data.filepath)
                    self.seg_table.set_select(0)
                self.seg_bar.pack(fill=tk.X, pady=1,expand=True)
                self.seg_table.pack(fill=tk.BOTH, expand=True)
        else:
            self.seg_bar.pack_forget()
            self.seg_table.pack_forget()
            self.file_bar.pack(fill=tk.X, pady=1,expand=True)
            self.file_table.pack(fill=tk.BOTH, expand=True)
        self.update_idletasks()

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
        filename:tuple[str,...]|Literal[''] = filedialog.askopenfilenames(filetypes=[("Stt files","*.npz"),("WAV files", "*.wav"),("all files", "*.*")])
        if isinstance(filename,tuple) and len(filename)>0:
            slist:list[SttData] = []
            for file_path in filename:
                if not os.path.isfile(file_path) or (not file_path.endswith('.npz') and not file_path.endswith('.wav')):
                    continue
                stt_data:SttData|None = SttData.load(file_path)
                if stt_data is None:
                    print(f"ロードできません: {file_path}")
                    continue
                if stt_data.typ != SttData.Text and stt_data.typ != SttData.Dump:
                    print(f"ロードできません: {file_path}")
                    return
                slist.append(stt_data)
            self._th_load_stt_list(slist)

    def load_dir(self):
        dirname:str = filedialog.askdirectory( initialdir=self._cwd, mustexist=True )
        if dirname:
            self._cwd = dirname
            try:
                if not os.path.isdir(dirname):
                    print(f"ロードできません: {dirname}")
                    return
                stt_list:list[SttData] = []
                for name in os.listdir(dirname):
                    stt_data:SttData|None = SttData.load( os.path.join(dirname,name) )
                    if stt_data:
                        stt_list.append(stt_data)
                if len(stt_list)==0:
                    return
                self._th_load_stt_list(stt_list)
            except:
                print(f"ロードできません: {dirname}")

    def _th_load_stt_list(self,stt_list:list[SttData]):
        try:
            if self.stt_engine is not None:
                self.stt_engine.stop()
                self.stt_engine = None
            self.plot(None)
            self.file_table.clear()
            if isinstance(stt_list,list|tuple):
                for stt_data in stt_list:
                    self._ev_queue.put( lambda stt_data=stt_data: self.file_table.add(stt_data, file_path=stt_data.filepath) )
                self._ev_queue.put( lambda: self.file_table.set_select(0) )
        except:
            print(f"ロードできません: {dir}")

    # 音声解析関数
    def analysis_audio(self):
        if self._current_stt_data is None or self._current_stt_data.filepath is None:
            return
        self.stt_engine = SttEngine( self._current_stt_data.filepath )

        butter0:tuple = self.stt_engine.get_audio_butter()
        butter:list = [v for v in butter0]
        for idx, ent in enumerate( [ self.fpass_entry, self.fstop_entry, self.gpass_entry, self.gstop_entry ] ):
            try:
                val = float(ent.get())
                butter[idx] = val
            except ValueError:
                ent.delete(0, tk.END)
                ent.insert(0, butter[idx] )
        #self.stt_engine.set_audio_butter( butter )
        self.stt_engine.start()
        while True:
            try:
                stt_data:SttData|None = self.stt_engine.get_data()
                self._ev_queue.put( lambda stt_data=stt_data,file_path=None: self.seg_table.add(stt_data, file_path=file_path) )
            except Empty:
                time.sleep(0.2)
                continue
            # ToDo

    def run_analysis(self):
        if self._current_stt_data is None or self._current_stt_data.filepath is None:
            print("ファイルが選択されていません")
            return
        self.plot(None)
        self.seg_table.clear()
        self.seg_table.add(self._current_stt_data, self._current_stt_data.filepath)
        t = Thread( target=self.analysis_audio, daemon=True )
        t.start()

    def update_result(self, stt_data:SttData):
        self._ev_queue.put( lambda stt_data=stt_data,file_path=None: self.seg_table.add(stt_data, file_path=file_path) )
        #self.table.add(stt_data)
        # self.plot(stt_data)

    def on_file_select(self, stt_data:SttData|None ):
        try:
            self._current_file = stt_data
            if isinstance(stt_data,SttData):
                filename_only = os.path.basename(stt_data.filepath or '')
                self.fn_text.configure(text=filename_only)
                self.plot(stt_data)
            else:
                self.fn_text.configure(text='')
                self.plot(None)
        except:
            traceback.print_exc()

    def on_seg_select(self, stt_data:SttData|None ):
        try:
            self.plot(stt_data if isinstance(stt_data,SttData) else None )
        except:
            traceback.print_exc()

    def plot(self,stt_data:SttData|None):
        self.plot1.set_stt_data(stt_data)

if __name__ == '__main__':
    app = Application()
    app.mainloop()

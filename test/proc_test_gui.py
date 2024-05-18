
from io import BytesIO
import traceback
from multiprocessing import Queue as PQ
from threading import Thread
from queue import Queue, Empty
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
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
from CrabAI.vmp import Ev, VFunction, VProcess
from CrabAI.voice._stt.proc_source_to_audio import SourceToAudio, shrink
from CrabAI.voice._stt.proc_source_to_audio import get_mic_devices
from CrabAI.voice._stt.proc_audio_to_segment import AudioToSegment
from CrabAI.voice._stt.proc_segment_to_voice import SegmentToVoice
from CrabAI.voice._stt.proc_voice_to_text import VoiceToText

def _getvalue(entry):
    try:
        return float(entry.get())
    except ValueError:
        return None

def _setvalue(ent,value):
    ent.delete(0, tk.END)
    ent.insert(0, value )


def show_mic_selection_dialog(root, samplerate, app):
    inp_dev_list = get_mic_devices(samplerate=samplerate, dtype='float32')
    if not inp_dev_list or len(inp_dev_list)<=0:
        messagebox.showerror("Error", "No input devices found")
        return None,None

    dialog = tk.Toplevel(root)
    dialog.title("Select Mic")
    
    listbox = tk.Listbox(dialog)
    listbox.pack(pady=20, padx=20)

    for dev in inp_dev_list:
        listbox.insert(tk.END, f"{dev['index']}: {dev['name']}")

    def on_select(dialog, listbox, inp_dev_list, app):
        selection = listbox.curselection()
        if selection:
            selected_index = selection[0]
            app.mic_index = inp_dev_list[selected_index]['index']
            app.mic_name = inp_dev_list[selected_index]['name']
            print(f"Selected Mic - Index: {app.mic_index}, Name: {app.mic_name}")
        else:
            print("No mic selected")
        dialog.destroy()

    select_button = tk.Button(dialog, text="Select", command=lambda: on_select(dialog, listbox, inp_dev_list, app))
    select_button.pack(pady=10)

NO_FILE="File..."
NO_MIC="Mic..."
MIC_NOT_FOUND="No input devices found"
BTN_RUN="Start"
BTN_STP="Stop"

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.mic_list = None
        self.title('音声解析GUI')
        self.geometry('800x600')
        self.create_widgets()

        self._ev_queue = Queue()
        self.running = True
        self.after_id=None
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # ウィンドウが閉じられたときのイベントハンドラを設定
        self._idle_loop()

        self.stt=None
        self.data_in1 = None
        self.data_in2 = None
        self.data_in3 = None
        self.data_in4 = None
        self.data_out = None
        self.ctl_out = None

    def create_widgets(self):

        # ボタンを配置するフレーム
        self.menu_bar = tk.Menu
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, pady=10)

        # ファイル選択ボタン
        self.load_button = ttk.Button(self.button_frame, text=NO_FILE, command=self.load_file)
        self.load_button.pack(side=tk.LEFT)

        self.mic_var = tk.StringVar(self.button_frame)
        self.mic_combobox = ttk.Combobox(self.button_frame, textvariable=self.mic_var)
        self.mic_combobox.set(NO_MIC)
        self.mic_combobox.bind("<Button-1>", self.update_mic_list)  # Comboboxをクリックしたときにリストを更新
        self.mic_combobox.bind("<<ComboboxSelected>>", self.on_mic_selected)  # Comboboxで項目が選択されたときのイベント
        self.mic_combobox.pack(side=tk.LEFT)

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
        self.run_button = ttk.Button(self.button_frame, text=BTN_RUN, command=self.run_analysis)
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
                self.load_button.configure(text=filename_only)
                self.mic_combobox.set(NO_MIC)
                self.mic_index = None
                self.mic_name = None
                self.plot(None)
                self.table.clear()
                self._ev_queue.put( lambda stt_data=stt_data,file_path=file_path: self.table.add(stt_data, file_path=file_path) )
        except:
            print(f"ロードできません: {file_path}")

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Stt files","*"),("WAV files", "*.wav"),("all files", "*")])
        if filename:
            self._th_load_from_directory(filename)

    def update_mic_list(self, event):
        inp_dev_list = self.mic_list
        if inp_dev_list is None:
            inp_dev_list = self.mic_list = get_mic_devices(samplerate=16000, dtype=np.float32)

        if inp_dev_list and len(inp_dev_list) > 0:
            mic_names = [f"{dev['index']}: {dev['name']}" for dev in inp_dev_list]
            self.mic_combobox['values'] = mic_names
        else:
            self.mic_combobox['values'] = [MIC_NOT_FOUND]

    def on_mic_selected(self, event):
        selected_mic = self.mic_var.get()
        if self.mic_list is None or selected_mic is None or MIC_NOT_FOUND in selected_mic:
            return

        selected_index = int(selected_mic.split(":")[0])
        self.mic_index = self.mic_list[selected_index]['index']
        self.mic_name = self.mic_list[selected_index]['name']
        self.load_button.configure(text=NO_FILE)
        self.filename = None
        #print(f"Selected Mic - Index: {self.mic_index}, Name: {self.mic_name}")

    # 音声解析関数
    def analysis_audio(self):

        if self.data_in1 is not None:
            return
        print("[TEST003] Test start")
        ctl_out = self.ctl_out = PQ()
        data_in1 = self.data_in1 = PQ()
        data_in2 = self.data_in2 = PQ()
        data_in3 = self.data_in3 = PQ()
        data_in4 = self.data_in4 = PQ()
        data_out= self.data_out = PQ()
        # 
        src=None
        mic=None
        if self.filename is not None:
            src = self.filename
        elif self.mic_index is not None:
            mic = self.mic_index
        wav_file = self.filename
        th1 = VProcess( SourceToAudio, data_in1, data_in2, ctl_out, sample_rate=16000, mic=mic, source=src )
        th2 = VProcess( AudioToSegment, data_in2, data_in3, ctl_out, sample_rate=16000 )
        th3 = VProcess( SegmentToVoice, data_in3, data_in4, ctl_out, sample_rate=16000 )
        th4 = VProcess( VoiceToText, data_in4, data_out, ctl_out )

        print("[TEST003] Process start")
        th4.start()
        th3.start()
        th2.start()
        th1.start()

        print("[TEST003] Loop")
        while True:
            try:
                stt_data:SttData = ctl_out.get( timeout=0.1 )
                if stt_data.typ == SttData.Dump:
                    print( f"[DUMP] {stt_data}")
                    self.update_result( stt_data )
            except Empty:
                pass

            try:
                stt_data:SttData = data_out.get( timeout=0.1 )
                print( f"[OUT] {stt_data}")
                if stt_data.typ == SttData.Text:
                    self.update_result( stt_data )
            except Empty:
                pass

            if th1.is_alive() or th2.is_alive() or th3.is_alive() or th4.is_alive():
                continue
            else:
                break
        self.data_in1 = None
        self.data_in2 = None
        self.data_in3 = None
        self.data_in4 = None
        self.data_out = None
        self.ctl_out = None
        self.run_button.configure( text=BTN_RUN)

    def stop_analysis(self):
        pass

    def run_analysis(self):
        if self.run_button.cget('text')==BTN_RUN:
            if hasattr(self, 'filename') or hasattr(self,'mic_index'):
                self.run_button.configure( text=BTN_STP)
                self.plot(None)
                self.table.clear()
                t = Thread( target=self.analysis_audio, daemon=True )
                t.start()
            else:
                print("ファイルが選択されていません")
        else:
            if self.data_in1 is not None:
                self.data_in1.put( Ev(0, Ev.Stop) )
                self.data_in2.put( Ev(0, Ev.Stop) )
                self.data_in3.put( Ev(0, Ev.Stop) )
                self.data_in4.put( Ev(0, Ev.Stop) )


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

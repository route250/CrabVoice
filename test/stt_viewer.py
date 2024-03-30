
import sys,os,glob
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import pygame
from scipy import signal
import sys,os
sys.path.append(os.getcwd())
from CrabAI.voice._stt.audio_to_text import SttData
from CrabAI.voice.voice_utils import audio_to_wave_bytes, audio_to_wave

def lowpass(x, samplerate, fpass, fstop, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fpass / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fstop / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y 
def hipass(x, samplerate, fpass, fstop, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fpass / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fstop / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "high")   #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)    #信号に対してフィルタをかける
    return y  
# SttDataを表示、再生するGUI

class SttDataViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('音声解析GUI')
        self.geometry('800x600')
        self.results = {}
        self.create_widgets()
        self.file_dir = '.'

    def create_widgets(self):

        # ボタンを配置するフレーム
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(fill=tk.X, pady=10)

        # ファイル選択ボタン
        self.load_button = tk.Button(self.button_frame, text='ファイルを選択', command=self.load_file)
        self.load_button.pack(side=tk.LEFT)

        # 再生ボタン
        self.play_button = tk.Button(self.button_frame, text='再生', command=self.play_audio)
        self.play_button.pack(side=tk.LEFT)

        # 停止ボタン
        self.stop_button = tk.Button(self.button_frame, text='停止', command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT)

        # FFTボタン
        self.stop_button = tk.Button(self.button_frame, text='FFT', command=self.show_fft_spectrum)
        self.stop_button.pack(side=tk.LEFT)

        # wave保存ボタン
        self.play_button = tk.Button(self.button_frame, text='Save', command=self.save_audio)
        self.play_button.pack(side=tk.LEFT)

        # 結果表示テーブル
        self.tree = ttk.Treeview(self, columns=('file','start', 'end', 'sec', 'sig','vad', 'content'), show='headings')
        self.tree.heading('file', text='ファイル名')
        self.tree.heading('start', text='開始フレーム')
        self.tree.heading('end', text='終了フレーム')
        self.tree.heading('sec', text='長さ')
        self.tree.heading('sig', text='sig')
        self.tree.heading('vad', text='vad')
        self.tree.heading('content', text='結果テキスト')
        # カラム幅の設定
        self.tree.column('file', width=100)      # ファイル名カラムの幅
        self.tree.column('start', width=30)      # 開始フレームカラムの幅
        self.tree.column('end', width=30)        # 終了フレームカラムの幅
        self.tree.column('sec', width=20)        # 長さカラムの幅
        self.tree.column('sig', width=20)        # Sigカラムの幅
        self.tree.column('vad', width=20)        # vadカラムの幅
        self.tree.column('content', width=200)   # 結果テキストカラムの幅

        self.tree.pack(fill=tk.BOTH, expand=True)

        # 音声波形グラフ
        self.figure = plt.Figure(figsize=(6,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 選択イベントのバインド
        self.tree.bind('<<TreeviewSelect>>', self.on_item_select)

        self.ax1 = None
        self.ax2 = None
        self.plot_scale = 1.0
        self.plot_xlim = None
        self._press_x = None

    def load_file(self):
        try:
            self.file_dir = filedialog.askdirectory( initialdir=self.file_dir )
            if os.path.isdir(self.file_dir):
                self.results={}
                for item in self.tree.get_children():
                    self.tree.delete(item)
                pattern = os.path.join( self.file_dir, 'audio*.npz')
                file_path_list = glob.glob(pattern)
                for file_path in file_path_list:
                    self.update_result(file_path)
        except:
            print(f"ロードできません")

    def update_result(self, file_path):
        try:
            stt_data:SttData = SttData.load(file_path)
        except:
            print(f"ロードできません: {file_path}")
        if stt_data.typ != SttData.Text and stt_data.typ != SttData.Dump:
            return
        file_name, _ = os.path.splitext(os.path.basename(file_path))
        row_id = len(self.results)  # 現在の行数をキーとする
        self.results[row_id] = file_path  # 結果を辞書に追加
        # 結果をテーブルに表示
        sec = (stt_data.end-stt_data.start)/stt_data.sample_rate
        sig=round( max(max(stt_data.hists['hi']),abs(min(stt_data.hists['lo'])) ), 3)
        vad=round( max(stt_data.hists['vad1']), 3)
        self.tree.insert('', tk.END, values=( file_name, stt_data.start, stt_data.end, sec, sig, vad, stt_data.content))

    def on_item_select(self, event):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        file_path = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        try:
            stt_data:SttData = SttData.load(file_path)
        except:
            print(f"ロードできません: {file_path}")
        self.plot(stt_data)

    def _on_mouse_wheel(self, event):
        # 拡大縮小の基準スケール
        # マウスイベントの位置を取得（データ座標）
        if  event.xdata is None:
            return  # データ座標が取得できない場合は何もしない

        if event.button == 'up':
            # 拡大
            self.plot_scale = min(10,self.plot_scale+0.2)
        elif event.button == 'down':
            # 縮小
            self.plot_scale = max(1.0, self.plot_scale - 0.2 )
        else:
            return  # その他のボタンは無視

        if 0.99<self.plot_scale<1.01:
            self.plot_scale = 1.0
            new_xlim = self.plot_xlim
        else:
            # 現在のズームレベルに基づいて新しい表示範囲を計算
            cur_xlim = self.ax1.get_xlim()
            # カーソルの相対位置
            x_rate = (cur_xlim[1]-event.xdata)/(cur_xlim[1]-cur_xlim[0])

            scaled_range = (self.plot_xlim[1]-self.plot_xlim[0])/self.plot_scale
            xmax = event.xdata + scaled_range * x_rate
            xmin = xmax - scaled_range
            if xmin<self.plot_xlim[0]:
                xmin = self.plot_xlim[0]
                xmax = xmin + scaled_range
            elif xmax>self.plot_xlim[1]:
                xmax = self.plot_xlim[1]
                xmin = xmax - scaled_range
            new_xlim = [xmin,xmax]

        # 両方の軸にズーム操作を適用
        self.ax1.set_xlim(new_xlim)
        self.ax2.set_xlim(new_xlim)

        self.canvas.draw()

    def _on_press(self, event):
        self._press_x = event.xdata

    def _on_release(self, event):
        self._press_x = None

    def _on_motion(self, event):
        if self._press_x is None or event.xdata is None or self.plot_scale<=1.0:
            return
        xmin,xmax = self.ax1.get_xlim()
        width = xmax-xmin
        dx = event.xdata - self._press_x
        if dx>0:
            xmin = xmin - dx
            if xmin < self.plot_xlim[0]:
                xmin = self.plot_xlim[0]
            xmax = xmin + width
        elif dx<0:
            xmax = xmax - dx
            if xmax > self.plot_xlim[1]:
                xmax = self.plot_xlim[1]
            xmin = xmax - width
        else:
            return

        self.ax1.set_xlim(xmin,xmax)
        self.ax2.set_xlim(xmin,xmax)
        self.canvas.draw()

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
            self.ax1 = self.figure.add_subplot(2, 1, 1)  # 上のサブプロット
            self.ax2 = self.figure.add_subplot(2, 1, 2)  # 下のサブプロット
            ax3 = self.ax2.twinx()

            x_sec = [ round((stt_data.start + (i*chunk_size))/stt_data.sample_rate,3) for i in range(len(hi)) ]
            self.plot_xlim = (x_sec[0],x_sec[-1])

            self.ax1.fill_between( x_sec, lo, hi, color='gray')
            self.ax1.set_ylim( ymin=-1.0, ymax=1.0)
            self.ax1.grid(True)

            self.ax2.plot( x_sec, vad, color='r', label='vad' )
            self.ax2.set_ylim( ymin=0.0, ymax=1.5 )
            self.ax2.grid(True)
            ax3.step( x_sec, co, where='post', color='b', label='color' )
            ax3.set_ylim( ymin=0.0, ymax=3.0 )

            h2, l2 = self.ax2.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            self.ax2.legend( h2+h3, l2+l3, loc='upper right')

            self.canvas.mpl_connect('scroll_event', self._on_mouse_wheel)
            self.canvas.mpl_connect('button_press_event', self._on_press)
            self.canvas.mpl_connect('button_release_event', self._on_release)
            self.canvas.mpl_connect('motion_notify_event', self._on_motion)

            self.canvas.draw()

    def play_audio(self):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        file_path = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        try:
            stt_data:SttData = SttData.load(file_path)
        except:
            print(f"ロードできません: {file_path}")
        st_sec, ed_sec = self.ax1.get_xlim()
        if stt_data is not None and stt_data.audio is not None:
            st = max(0, int(st_sec * stt_data.sample_rate) - stt_data.start)
            ed = min( len(stt_data.audio), int(ed_sec * stt_data.sample_rate) - stt_data.start )
            bb = audio_to_wave_bytes( stt_data.audio[st:ed], sample_rate=stt_data.sample_rate)
            xx = BytesIO(bb)
            pygame.mixer.init()
            pygame.mixer.music.load(xx)
            pygame.mixer.music.play()
        else:
            print("ファイルが選択されていません")

    def save_audio(self):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        file_path = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        try:
            stt_data:SttData = SttData.load(file_path)
        except:
            print(f"ロードできません: {file_path}")
        st_sec, ed_sec = self.ax1.get_xlim()
        if stt_data is not None and stt_data.audio is not None:
            file_name, _ = os.path.splitext(os.path.basename(file_path))
            files = [('Wave Files', '*.wav'),('All Files', '*.*')]  
            out = filedialog.asksaveasfilename( filetypes=files, initialdir=self.file_dir, initialfile=file_name, confirmoverwrite=True, defaultextension=files )
            st = max(0, int(st_sec * stt_data.sample_rate) - stt_data.start)
            ed = min( len(stt_data.audio), int(ed_sec * stt_data.sample_rate) - stt_data.start )
            audio_to_wave( out, stt_data.audio[st:ed], samplerate=stt_data.sample_rate)
        else:
            print("ファイルが選択されていません")

    def stop_audio(self):
        pygame.mixer.music.stop()

    def show_fft_spectrum(self):
        selected_item = self.tree.selection()[0]  # 選択されたアイテムID
        row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
        file_path = self.results.get(row_id)  # 対応するSTTDataオブジェクトを取得
        try:
            stt_data:SttData = SttData.load(file_path)
        except:
            print(f"ロードできません: {file_path}")
        if stt_data is None or stt_data.audio is None:
            print("ファイルが選択されていません")
            return
        # FFTの実行
        st_sec, ed_sec = self.ax1.get_xlim()
        st = max(0, int(st_sec * stt_data.sample_rate) - stt_data.start)
        ed = min( len(stt_data.audio), int(ed_sec * stt_data.sample_rate) - stt_data.start )
        audio_raw:np.ndarray = stt_data.audio[st:ed]
        N = len(audio_raw)
        T = 1.0 / stt_data.sample_rate
        # ハミングウィンドウを適用
        window = np.hamming(N)
        audio_windowed = audio_raw * window
        yf = np.fft.fft(audio_windowed)
        xf = np.fft.fftfreq(N, T)[:N//2]
        # ローカット
        audio_cut = hipass( audio_raw, stt_data.sample_rate, fpass=50, fstop=10, gpass=1, gstop=5)
        audio_cut = audio_cut * window
        yf_cut = np.fft.fft(audio_cut)
        xf_cut = np.fft.fftfreq(N, T)[:N//2]

        # 新しいFigureを作成してスペクトラムをプロット
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]),color='r')
        ax.plot(xf_cut, 2.0/N * np.abs(yf_cut[:N//2]),color='b')
        ax.set_title('FFT Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|Y(f)|')

        # 新しいウィンドウでプロットを表示
        new_window = tk.Toplevel(self)
        new_window.title("FFT Spectrum")
        # スケール選択用のComboboxを作成
        scale_options = ['Linear', 'Log']
        scale_var = tk.StringVar()
        scale_combobox = ttk.Combobox(new_window, textvariable=scale_var, values=scale_options, state='readonly')
        scale_combobox.pack()
        scale_combobox.set('Linear')  # デフォルト値を設定
        # Comboboxの選択値が変更されたときに実行するコールバック関数
        def on_scale_changed(event=None):
            selected_scale = scale_var.get()
            if selected_scale == 'Log':
                ax.set_xscale('log')
            else:
                ax.set_xscale('linear')
            canvas.draw()

        # Comboboxの選択イベントにコールバック関数をバインド
        scale_combobox.bind('<<ComboboxSelected>>', on_scale_changed)

        # 最大周波数のスライダーバー
        max_freq_scale = tk.Scale(new_window, from_=0, to=stt_data.sample_rate/2, orient='horizontal', label='Max Frequency (Hz)', resolution=100)
        max_freq_scale.set(stt_data.sample_rate/2)  # 最大値を初期値として設定
        max_freq_scale.pack(fill='x')

        def update_spectrum(event=None):
            max_freq = max_freq_scale.get()
            ax.set_xlim(0, max_freq)
            canvas.draw()

        # スライダーの値が変更されたときにスペクトラムを更新
        max_freq_scale.configure(command=update_spectrum)

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == '__main__':
    app = SttDataViewer()
    app.mainloop()

import sys,os
import time
import numpy as np
from scipy import signal
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
sys.path.append(os.getcwd())
from CrabAI.voice._stt.audio_to_text import SttData

class SttDataPlotter:

    def __init__(self, parent):

        self.figure = plt.Figure(figsize=(6, 2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.ax1 = None
        self.plot_scale = 1.0
        self.plot_xlim = None
        self._press_x = None
        self._update_time:float = 0

    def pack(self,*args,**kwargs):
        self.canvas.get_tk_widget().pack( *args, **kwargs )

    def plot(self, stt_data: SttData):
        self.figure.clear()
        if stt_data is not None and stt_data.hists is not None:
            hists = stt_data.hists
            hi = hists['hi']
            lo = hists['lo']
            co = hists['color']
            vad = hists['vad1']
            var = hists['var']

            frames = stt_data.end - stt_data.start
            chunks = len(hi)
            chunk_size = frames // chunks

            ymin=0.0
            ymax=1.5

            ymid = (ymax-ymin)/2
            yhalf = ymax-ymid
            hi = hi * yhalf + ymid
            lo = lo * yhalf + ymid

            self.ax1 = self.figure.add_subplot(1, 1, 1)
            ax3 = self.ax1.twinx()

            x_sec = [round((stt_data.start + (i * chunk_size)) / stt_data.sample_rate, 3) for i in range(len(hi))]
            self.plot_xlim = (x_sec[0], x_sec[-1])

            self.ax1.fill_between(x_sec, lo, hi, color='gray')

            self.ax1.plot(x_sec, vad, color='r', label='vad')
            self.ax1.plot(x_sec, var, color='y', label='var')
            self.ax1.set_ylim(ymin=ymin, ymax=ymax)
            self.ax1.grid(True)
            ax3.step(x_sec, co, where='post', color='b', label='color')
            ax3.set_ylim(ymin=0, ymax=3.0)

            h2, l2 = self.ax1.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            self.ax1.legend(h2 + h3, l2 + l3, loc='upper right')

            self.canvas.mpl_connect('scroll_event', self._on_mouse_wheel)
            self.canvas.mpl_connect('button_press_event', self._on_press)
            self.canvas.mpl_connect('button_release_event', self._on_release)
            self.canvas.mpl_connect('motion_notify_event', self._on_motion)

            self.canvas.draw()

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
        self.ax1.set_xlim(new_xlim)

        self.canvas.draw()

    def _on_press(self, event):
        self._press_x = event.xdata
        self._update_time = time.time()

    def _on_release(self, event):
        self._press_x = None

    def _on_motion(self, event):
        if self._press_x is None or event.xdata is None or self.plot_scale<=1.0:
            return
        now = time.time()
        if (now-self._update_time)<0.2:
            return
        self._update_time = now
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
        self.ax1.set_xlim(xmin,xmax)
        self.canvas.draw()

    def get_xlim(self):
        return self.ax1.get_xlim()

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

class FFTplot:

    def __init__(self,parent, raw:np.ndarray, audio:np.ndarray, sampling_rate:int ):

        self._sampling_rate:int = sampling_rate
        self._raw:np.ndarray = raw
        self._raw_x = None
        self._raw_y = None
        self._audio:np.ndarray = audio
        self._audio_x = None
        self._audio_y = None
        self._cut_x = None
        self._cut_y = None

        self.parent = parent
        self.frame = ttk.Frame( parent )
        toolbar = ttk.Frame( self.frame )
        toolbar.pack( side=tk.TOP )

        # スケール選択用のComboboxを作成
        scale_options = ['Linear', 'Log']
        self._scale_var = tk.StringVar()
        scale_combobox = ttk.Combobox( toolbar, textvariable=self._scale_var, values=scale_options, state='readonly' )
        scale_combobox.pack( side=tk.LEFT, padx=10 )
        scale_combobox.set('Linear')  # デフォルト値を設定
        scale_combobox.bind ( " <<ComboboxSelected>> " , self._on_scale_changed )
        # 最大周波数のスライダーバー
        self.prev_max_freq=None
        self.prev_scale=None
        default_freq = self._sampling_rate//2
        self.max_freq_label = ttk.Label(toolbar, text=f'Max Freq.(Hz): {default_freq}')
        self.max_freq_label.pack( side=tk.LEFT, padx=4 )
        self.max_freq_scale = ttk.Scale( toolbar, from_=0, to=default_freq, value=default_freq, orient=tk.HORIZONTAL, command=self._on_scale_changed)
        self.max_freq_scale.pack( side=tk.LEFT, padx=4 )

        self.figure = plt.Figure(figsize=(3, 2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame )
        self.canvas.get_tk_widget().pack( side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax1 = self.figure.add_subplot(1, 1, 1)

        self.ck1_var = tk.BooleanVar(value=raw is not None)
        self.ck2_var = tk.BooleanVar(value=audio is not None)
        self.ck3_var = tk.BooleanVar(value=False)
        self.ck1 = ttk.Checkbutton( toolbar, text='raw', variable=self.ck1_var, command=self._on_sw_changed )
        self.ck1.pack( side=tk.LEFT, padx=4 )
        self.ck2 = ttk.Checkbutton( toolbar, text='audio', variable=self.ck2_var, command=self._on_sw_changed )
        self.ck2.pack( side=tk.LEFT, padx=4 )
        self.ck3 = ttk.Checkbutton( toolbar, text='fft', variable=self.ck3_var, command=self._on_sw_changed )
        self.ck3.pack( side=tk.LEFT, padx=4 )
        self.prev_ck1=None
        self.prev_ck2=None
        self.prev_ck3=None

    def pack(self, *args, **kwargs ):
        self.frame.pack( *args, **kwargs )
        self._plot()

    def _on_scale_changed(self,*args):
        self.frame.after_idle( self._redraw )

    def _on_sw_changed(self,*args):
        self._plot()

    def _redraw(self):
        freq = max( 100, (self.max_freq_scale.get()//100 ) * 100 )
        selected_scale = self._scale_var.get()
        if freq != self.prev_max_freq or selected_scale != self.prev_scale:
            self.prev_max_freq = freq
            self.prev_scale = selected_scale
            self.max_freq_label.config(text=f"Max Freq. {freq}Hz")
            if selected_scale == 'Log':
                self.ax1.set_xscale('log')
                self.ax1.set_xlim(1, freq)
            else:
                self.ax1.set_xscale('linear')
                self.ax1.set_xlim(0, freq)
            self.canvas.draw_idle()

    # def _filter(self):
    #     audio_cut = hipass( audio_raw, stt_data.sample_rate, fpass=50, fstop=10, gpass=1, gstop=5)
    #     audio_cut = audio_cut * window
    #     yf_cut = np.fft.fft(audio_cut)
    #     xf_cut = np.fft.fftfreq(N, T)[:N//2]
    
    def _fft(self):
        if self._raw_y is None or self._raw_x is None:
            N = len(self._raw) if self._raw is not None else len(self._audio)
            T = 1.0 / self._sampling_rate
            # ハミングウィンドウを適用
            window = np.hamming(N)
            if self._raw is not None:
                raw = self._raw * window
                self._raw_y = np.fft.fft(raw)
                self._raw_x = np.fft.fftfreq(N, T)[:N//2]

            if self._audio is not None:
                audio = self._audio * window
                self._audio_y = np.fft.fft(audio)
                self._audio_x = np.fft.fftfreq(N, T)[:N//2]
    
    def _fft2(self):
        if self._cut_x is None or self._cut_y is None:
            audio_cut = self._raw if self._raw is not None else self._audio
            if audio_cut is not None:
                audio_cut = hipass( audio_cut, self._sampling_rate, fpass=50, fstop=10, gpass=1, gstop=5)
                N = len(audio_cut)
                T = 1.0 / self._sampling_rate
                # ハミングウィンドウを適用
                window = np.hamming(N)
                audio_cut = audio_cut * window
                self._cut_y = np.fft.fft(audio_cut)
                self._cut_x = np.fft.fftfreq(N, T)[:N//2]

    def _plot(self):
        self._fft()
        self.figure.clear()

        N = len(self._raw) if self._raw is not None else len(self._audio)
        self.ax1 = self.figure.add_subplot(1,1,1)
        if self.ck1_var.get() and self._raw_x is not None and self._raw_y is not None:
            self.ax1.plot( self._raw_x, 2.0/N * np.abs(self._raw_y[:N//2]),color='r', label='raw')
        if self.ck2_var.get() and self._audio_x is not None and self._audio_y is not None:
            self.ax1.plot(self._audio_x, 2.0/N * np.abs(self._audio_y[:N//2]),color='b', label='audio')
        if self.ck3_var.get():
            self._fft2()
            if self._cut_x is not None and self._cut_y is not None:
                self.ax1.plot(self._cut_x, 2.0/N * np.abs(self._cut_y[:N//2]),color='g',label='cut')
        self.ax1.set_title('FFT Spectrum')
        self.ax1.set_xlabel('Frequency (Hz)')
        self.ax1.set_ylabel('|Y(f)|')

        self.canvas.draw()

class SttDataTable:

    def __init__(self,parent):

        self.stt_data_map = {}
        self.file_path_map = {}

        # Frameウィジェットを作成して、Treeviewとスクロールバーを格納
        self.frame = ttk.Frame(parent)
        self.tree = ttk.Treeview(self.frame, columns=('file','utc', 'start', 'end', 'sec', 'sig','vad', 'content'), show='headings')
        # 縦スクロールバーの作成
        self.v_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.v_scroll.set)

        self.tree.heading('file', text='ファイル名')
        self.tree.heading('utc', text='utc')
        self.tree.heading('start', text='開始フレーム')
        self.tree.heading('end', text='終了フレーム')
        self.tree.heading('sec', text='長さ')
        self.tree.heading('sig', text='sig')
        self.tree.heading('vad', text='vad')
        self.tree.heading('content', text='結果テキスト')
        # カラム幅の設定
        self.tree.column('file', width=100)      # ファイル名カラムの幅
        self.tree.column('utc', width=30)
        self.tree.column('start', width=30)      # 開始フレームカラムの幅
        self.tree.column('end', width=30)        # 終了フレームカラムの幅
        self.tree.column('sec', width=20)        # 長さカラムの幅
        self.tree.column('sig', width=20)        # Sigカラムの幅
        self.tree.column('vad', width=20)        # vadカラムの幅
        self.tree.column('content', width=200)   # 結果テキストカラムの幅
        # 選択イベントのバインド
        self._ev_on_select = None
        self.tree.bind('<<TreeviewSelect>>', self._fn_on_select)

    def pack(self,*args,**kwargs):
        self.frame.pack( *args, **kwargs )
        self.tree.pack( side=tk.LEFT, fill=tk.BOTH, expand=True )
        self.v_scroll.pack(side=tk.RIGHT, fill="y")

    def bind(self, fn ):
        self._ev_on_select = fn

    def _fn_on_select(self, event):
        if self._ev_on_select is not None:
            selected_item = self.tree.selection()[0]  # 選択されたアイテムID
            row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
            stt_data:SttData = self.stt_data_map.get(row_id)  # 対応するSTTDataオブジェクトを取得
            if stt_data is None:
                file_path = self.file_path_map.get(row_id)  # 対応するSTTDataオブジェクトを取得
                try:
                    stt_data:SttData = SttData.load(file_path)
                except:
                    print(f"ロードできません: {file_path}")
                    return
            self._ev_on_select(stt_data)

    def clear(self):
        self.stt_data_map={}
        self.file_path_map={}
        self.tree.selection_clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

    def selection_clear(self):
        self.tree.selection_clear()

    def load(self, file_path):
        try:
            stt_data:SttData = SttData.load(file_path)
            if stt_data is None:
                print(f"ロードできません: {file_path}")
                return
        except:
            print(f"ロードできません: {file_path}")

    def add(self, stt_data:SttData, file_path=None ):
        if stt_data.typ != SttData.Text and stt_data.typ != SttData.Dump:
            return
        file_name, _ = os.path.splitext(os.path.basename(file_path)) if file_path is not None else None,None
        row_id = len(self.file_path_map)  # 現在の行数をキーとする
        self.stt_data_map[row_id] = stt_data  # 結果を辞書に追加
        self.file_path_map[row_id] = file_path  # 結果を辞書に追加
        # 結果をテーブルに表示
        sec = (stt_data.end-stt_data.start)/stt_data.sample_rate
        sig=round( max(max(stt_data.hists['hi']),abs(min(stt_data.hists['lo'])) ), 3)
        vad=round( max(stt_data.hists['vad1']), 3)
        self.tree.insert('', tk.END, values=( file_name, stt_data.utc, stt_data.start, stt_data.end, sec, sig, vad, stt_data.content))

    def selected(self) ->SttData:
        try:
            selected_item = self.tree.selection()[0]  # 選択されたアイテムID
            row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
            file_path = self.file_path_map.get(row_id)  # 対応するSTTDataオブジェクトを取得
            stt_data:SttData= self.stt_data_map.get(row_id)  # 対応するSTTDataオブジェクトを取得
            return stt_data
        except:
            print(f"ロードできません: {file_path}")

    def selected_filepath(self) ->str:
        try:
            selected_item = self.tree.selection()[0]  # 選択されたアイテムID
            row_id = int(selected_item.split('I')[1], 16) - 1  # 行番号を取得
            file_path = self.file_path_map.get(row_id)  # 対応するSTTDataオブジェクトを取得
            return file_path
        except:
            print(f"ロードできません: {file_path}")

def test():
    testfile='logs/audio/20240330/audio_20240330_002635.npz'
    stt_data = SttData.load(testfile)
    
    raw:np.ndarray = stt_data.raw
    audio:np.ndarray = stt_data.audio

    root = tk.Tk()
    fftplot = FFTplot(root,raw,audio,stt_data.sample_rate)
    fftplot.pack()

    root.mainloop()

if __name__ == '__main__':
    test()
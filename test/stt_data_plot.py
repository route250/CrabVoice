import sys,os
from io import BytesIO
import time
import numpy as np
import pygame
from scipy import signal
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
sys.path.append(os.getcwd())
from CrabAI.voice._stt.audio_to_text import SttData
from CrabAI.voice.voice_utils import audio_to_wave_bytes, audio_to_wave

class SttDataPlotter(ttk.Frame):

    def __init__(self, parent, **kwargs ):
        super().__init__( parent, **kwargs )

        self._stt_data: SttData = None

        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 0.1

        self.toolbar = ttk.Frame( self )
        self.toolbar.pack( side=tk.TOP, fill=tk.X )

        # FFTボタン
        self.fft_button = ttk.Button(self.toolbar, text='FFT', command=self.show_fft_spectrum)
        self.fft_button.pack(side=tk.LEFT)

        # 再生ボタン
        self.play_button = ttk.Button(self.toolbar, text='再生(audio)', command=self.play_audio)
        self.play_button.pack(side=tk.LEFT)

        # 再生ボタン
        self.play_button = ttk.Button(self.toolbar, text='再生(raw)', command=self.play_raw)
        self.play_button.pack(side=tk.LEFT)

        # 停止ボタン
        self.stop_button = ttk.Button(self.toolbar, text='停止', command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT)

        # wave保存ボタン
        self.save_button = ttk.Button(self.toolbar, text='Save', command=self.save_audio)
        self.save_button.pack(side=tk.LEFT)

        self.btn = ttk.Button( self.toolbar)
        self.btn.pack( side=tk.LEFT )

        self.sidebar = ttk.Frame( self )
        self.sidebar.pack( side=tk.LEFT, fill=tk.X )

        self.btn2 = ttk.Button( self.sidebar)
        self.btn2.pack( side=tk.TOP )
        # サイドバー内のテキスト表示用ラベルを追加
        self.display_label = ttk.Label(self.sidebar, text="")
        self.display_label.pack(side=tk.TOP)

        self.plotframe = ttk.Frame( self )
        self.plotframe.pack( side=tk.BOTTOM, fill=tk.BOTH, expand=True )

        self.figure = plt.Figure(figsize=(6, 2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plotframe)
        self.canvas.get_tk_widget().pack( side=tk.TOP, fill=tk.BOTH, expand=True )

        self.ax1 = None
        self.ax3 = None
        self.x_sec = None
        self.x_frame = None
        self.y_hi = None
        self.y_lo = None
        self.y_color = None
        self.y_vad = None
        self.y_vad_ave = None
        self.y_vad_slope = None
        self.y_vad_accel = None
        self.y_var = None
        self.y_color = None
        self.plot_scale = 1.0
        self.plot_xlim = None
        self._press_x = None
        self._update_time:float = 0

        self.y_min = 0.0
        self.y_max = 1.5

    def set_stt_data(self, stt_data: SttData):

        self._stt_Data: SttData = stt_data
        self.ax1 = None
        self.ax3 = None
        self.x_sec = None
        self.x_frame = None
        self.y_hi = None
        self.y_lo = None
        self.y_color = None
        self.y_vad = None
        self.y_vad_ave = None
        self.y_vad_slope = None
        self.y_vad_accel = None
        self.y_var = None
        self.y_color = None
        self.plot_scale = 1.0
        self.plot_xlim = None
        self._press_x = None

        if stt_data is None or stt_data.hists is None:
            self._plot_draw()
            return

        hists = stt_data.hists
        hi = hists['hi']
        lo = hists['lo']
        if hi is None or lo is None or len(hi)!=len(lo) or len(hi)<2:
            self._plot_draw()
            return

        samples = stt_data.end - stt_data.start
        frames = len(hi)
        frame_size = samples // frames

        ymid = (self.y_max-self.y_min)/2
        yhalf = self.y_max-ymid
        hi = hi * yhalf + ymid
        lo = lo * yhalf + ymid
        self.y_hi = hi
        self.y_lo = lo

        co = hists['color']
        self.y_color = co
        vad = hists['vad']
        self.y_vad = vad

        vad_ave = stt_data['vad_ave']
        self.y_vad_ave = vad_ave

        vad_slope = stt_data['vad_slope']
        self.y_vad_slope = ( (vad_slope * 2 ) + 0.5 ) if vad_slope is not None else None

        vad_accel = stt_data['vad_accel']
        self.y_vad_accel = ( (vad_accel * 2 ) + 0.5 ) if vad_accel is not None else None

        var = hists['var']
        self.y_var = var

        x_frame = [ int(stt_data.start + (i * frame_size)) for i in range(len(hi))]
        self.x_frame = x_frame
        x_sec = [round( x_frame[i] / stt_data.sample_rate, 3) for i in range(len(hi))]
        self.x_sec = x_sec
        self.plot_xlim = (x_sec[0], x_sec[-1])

        self._plot_draw()

    def _plot_draw(self):

        x_sec = self.x_sec
        hi = self.y_hi
        lo = self.y_lo
        vad = self.y_vad
        vad_ave = self.y_vad_ave
        vad_slope = self.y_vad_slope
        vad_accel = self.y_vad_accel
        co = self.y_color
        var = self.y_var

        self.figure.clear() 
        self.ax1 = None
        self.ax3 = None
        self._update_time:float = 0

        if x_sec is None or hi is None or lo is None:
            self.canvas.draw_idle()
            return

        self.ax1 = self.figure.add_subplot(1, 1, 1)
        ax3 = self.ax1.twinx()
        self.ax3 = ax3
        ax3.set_ylim(ymin=0, ymax=15)
        ax3.set_yticks( [0,1,2,3,4,5,6,7,8,9,10])
        ax3.set_yticklabels( ['','1','2','3','4','5','6','7','8','9',''])
        self.ax1.set_ylim(ymin=self.y_min, ymax=self.y_max)
        self.ax1.set_yticks( [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] )
        self.ax1.set_yticklabels( ['0','','','','','.5','','','','','1.0'])

        self.ax1.fill_between(x_sec, lo, hi, color='gray')

        self.ax1.plot(x_sec, vad, color='r', label='vad')
        if vad_ave is not None:
            self.ax1.plot(x_sec, vad_ave, color='r', linestyle='--', label='vad_ave')
        if vad_slope is not None:
            self.ax1.plot(x_sec, vad_slope, color='y', label='vad_slope')
        if vad_accel is not None:
            self.ax1.plot(x_sec, vad_accel, color='y', linestyle='--', label='vad_accel')
        self.ax1.plot(x_sec, var, color='g', label='var')
        self.ax1.grid(True)
        ax3.step(x_sec, co, where='post', color='b', label='color')

        h2, l2 = self.ax1.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        self.ax1.legend(h2 + h3, l2 + l3, loc='upper right',ncol=4)

        self.canvas.mpl_connect('scroll_event', self._on_mouse_wheel)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self.canvas.draw_idle()

    def _on_mouse_wheel(self, event):
        # 拡大縮小の基準スケール
        # マウスイベントの位置を取得（データ座標）
        if  event.xdata is None:
            return  # データ座標が取得できない場合は何もしない

        if event.button == 'up':
            # 拡大
            self.plot_scale = min(100,self.plot_scale+0.2)
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
       # self.ax1.set_xlim(new_xlim)

        self.canvas.draw_idle()

    def _get_value(self, key, index, default=None ):
        if self._stt_Data is not None:
            y = self._stt_Data[key]
            if y is not None:
                return y[index]
        return default

    def _on_press(self, event):

        if event.inaxes != self.ax1 and event.inaxes != self.ax3:  # クリックされた場所がプロットエリア外であれば何もしない
            return

        self._press_x = event.xdata
        self._update_time = time.time()

        # クリックされたX座標に最も近いデータポイントのインデックスを探索
        index = np.abs(np.array(self.x_sec) - self._press_x).argmin()
        sec = self.x_sec[index]
        frame = self.x_frame[index]

        # 対応するVAD、VAD_AVE、VAD_SLOPEの値を取得
        y = self._stt_Data['vad']
        val_vad = self._get_value('vad',index,'N/A')
        val_vad_ave = self._get_value('vad_ave',index,'N/A')
        val_vad_slope = self._get_value('vad_slope',index,'N/A')
        val_vad_accel = self._get_value('vad_accel',index,'N/A')

        # 前回の垂直線とテキストがあれば消去
        if hasattr(self, '_vline'):
            self._vline.remove()

        # Y軸方向に半透明の垂直線を引く
        self._vline = self.ax1.axvline(x=sec, color='blue', linestyle='--', linewidth=2, alpha=0.5)
        self.canvas.draw_idle()

        # サイドバーのラベルにテキストを設定
        display_text = f"Sec: {sec:12.3f}\nFrame: {frame:12}\nVAD: {val_vad:12.2f}\nAVE: {val_vad_ave:12.2f}\nSLOPE: {val_vad_slope:12.2f}\nAccel:{val_vad_accel:12.2f}"
        self.display_label.config(text=display_text)

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
        #self.ax1.set_xlim(xmin,xmax)
        self.canvas.draw()

    def get_xlim(self):
        return self.ax1.get_xlim()

    def show_fft_spectrum(self):
        stt_data:SttData = self._stt_Data
        if stt_data is None or stt_data.audio is None:
            print("ファイルが選択されていません")
            return
        # FFTの実行
        st_sec, ed_sec = self.ax1.get_xlim()
        st = max(0, int(st_sec * stt_data.sample_rate) - stt_data.start)
        ed = min( len(stt_data.audio), int(ed_sec * stt_data.sample_rate) - stt_data.start )

        raw:np.ndarray = stt_data.raw[st:ed] if stt_data.raw is not None else None
        audio:np.ndarray = stt_data.audio[st:ed] if stt_data.audio is not None else None

        # 新しいウィンドウでプロットを表示
        root = self
        while not isinstance(root,tk.Tk):
            try:
                if root.master is None:
                    break
                root = root.master
            except:
                break
        new_window = tk.Toplevel(root)
        new_window.title("FFT Spectrum")
        new_window.geometry('800x600')
        fftplt = FFTplot(new_window, raw, audio, stt_data.sample_rate)
        fftplt.pack( side=tk.TOP, fill='both', expand=True)

    def play_audio(self):
        self.play_audiox(False)

    def play_raw(self):
        self.play_audiox(True)

    def play_audiox(self,b):
        stt_data:SttData = self._stt_Data
        if stt_data is None:
            print("stt_data is None")
            return
        st_sec, ed_sec = self.get_xlim()
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

    def save_audio(self):
        stt_data:SttData = self._stt_Data
        file_path = stt_data.filepath
        if stt_data is not None and stt_data.audio is not None:
            st_sec, ed_sec = self.get_xlim()
            file_name, _ = os.path.splitext(os.path.basename(file_path)) if file_path is not None else None
            files = [('Wave Files', '*.wav'),('All Files', '*.*')]  
            out = filedialog.asksaveasfilename( filetypes=files, initialdir=self.dir_path, initialfile=file_name, confirmoverwrite=True, defaultextension=files )
            st = max(0, int(st_sec * stt_data.sample_rate) - stt_data.start)
            ed = min( len(stt_data.audio), int(ed_sec * stt_data.sample_rate) - stt_data.start )
            audio_to_wave( out, stt_data.audio[st:ed], samplerate=stt_data.sample_rate)
        else:
            print("ファイルが選択されていません")

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

class FFTplot(ttk.Frame):

    def __init__(self,parent, raw:np.ndarray, audio:np.ndarray, sampling_rate:int ):
        super().__init__(parent)
        self._sampling_rate:int = sampling_rate
        self._raw:np.ndarray = raw
        self._raw_x = None
        self._raw_y = None
        self._audio:np.ndarray = audio
        self._audio_x = None
        self._audio_y = None
        self._cut_x = None
        self._cut_y = None

        self.frame = ttk.Frame( self )
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

        self.frame.pack( fill=tk.BOTH, expand=True )
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

class SttDataTable(ttk.Frame):

    def __init__(self, parent, **kwargs ):
        super().__init__( parent, **kwargs )

        self.stt_data_map = {}
        self.file_path_map = {}

        # Frameウィジェットを作成して、Treeviewとスクロールバーを格納
        self.tree:ttk.Treeview = ttk.Treeview(self, columns=('file','utc', 'typ', 'start', 'end', 'sec', 'sig','vad', 'content'), show='headings')
        # 縦スクロールバーの作成
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.v_scroll.set)

        self.tree.heading('file', text='ファイル名')
        self.tree.heading('utc', text='utc')
        self.tree.heading('typ', text='type')
        self.tree.heading('start', text='開始フレーム')
        self.tree.heading('end', text='終了フレーム')
        self.tree.heading('sec', text='長さ')
        self.tree.heading('sig', text='sig')
        self.tree.heading('vad', text='vad')
        self.tree.heading('content', text='結果テキスト')
        # カラム幅の設定
        self.tree.column('file', width=100)      # ファイル名カラムの幅
        self.tree.column('utc', width=30)
        self.tree.column('typ', width=30)
        self.tree.column('start', width=30)      # 開始フレームカラムの幅
        self.tree.column('end', width=30)        # 終了フレームカラムの幅
        self.tree.column('sec', width=20)        # 長さカラムの幅
        self.tree.column('sig', width=20)        # Sigカラムの幅
        self.tree.column('vad', width=20)        # vadカラムの幅
        self.tree.column('content', width=200)   # 結果テキストカラムの幅
        # 選択イベントのバインド
        self._ev_on_select = None
        self.tree.bind('<<TreeviewSelect>>', self._fn_on_select)

        self.tree.pack( side=tk.LEFT, fill=tk.BOTH, expand=True )
        self.v_scroll.pack(side=tk.RIGHT, fill="y")

    def bind(self, fn ):
        self._ev_on_select = fn

    def selection_clear(self):
        self.tree.selection_clear()

    def _selected_iid(self):
        try:
            return self.tree.selection()[0]
        except:
            return None

    def selected(self) ->SttData:
        try:
            iid = self._selected_iid() # 選択されたアイテムID
            file_path = self.file_path_map.get(iid)  # 対応するSTTDataオブジェクトを取得
            stt_data:SttData= self.stt_data_map.get(iid)  # 対応するSTTDataオブジェクトを取得
            if stt_data is not None:
                return stt_data
            if file_path is not None:
                stt_data:SttData = SttData.load(file_path)
                if stt_data is not None:
                    return stt_data
                print(f"ロードできません: {file_path}")
        except:
            print(f"ロードできません: {file_path}")

    def _fn_on_select(self, event):
        if self._ev_on_select is not None:
            stt_data:SttData = self.selected()
            self._ev_on_select(stt_data)

    def clear(self):
        self._fn_on_select(None)
        self.stt_data_map={}
        self.file_path_map={}
        self.tree.selection_clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

    def add(self, stt_data:SttData, file_path=None ):
        if stt_data.typ != SttData.Text and stt_data.typ != SttData.Dump:
            return
        file_name, _ = os.path.splitext(os.path.basename(file_path)) if file_path is not None else None,None
        # 結果をテーブルに表示
        sec = (stt_data.end-stt_data.start)/stt_data.sample_rate
        sig=round( max(max(stt_data.hists['hi']),abs(min(stt_data.hists['lo'])) ), 3)
        vad=round( max(stt_data.hists['vad']), 3)
        typ = SttData.type_to_str(stt_data.typ)
        values=( file_name, stt_data.utc, typ, stt_data.start, stt_data.end, sec, sig, vad, stt_data.content)
        # 挿入位置を見つける
        children = self.tree.get_children()
        insert_index = 0
        for i, child in enumerate(children):
            item = self.tree.item(child)
            if stt_data.start < int(item['values'][3]):
                insert_index = i
                break
            insert_index = i + 1
        iid:str = self.tree.insert('', insert_index, values=values)
        self.stt_data_map[iid] = stt_data  # 結果を辞書に追加
        self.file_path_map[iid] = file_path  # 結果を辞書に追加

    def selected_filepath(self) ->str:
        try:
            iid = self._selected_iid() # 選択されたアイテムID
            file_path = self.file_path_map.get(iid)  # 対応するSTTDataオブジェクトを取得
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
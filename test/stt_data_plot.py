import sys,os
import time
import tkinter as tk
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
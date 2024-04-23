import numpy as np
import pandas as pd
from .ring_buffer import RingBuffer

class AudioFeatureBuffer:

    def __init__(self,capacity,window:int=5):
        self.window:int = ( window//2 )*2 + 1 if window>=3 else 3
        self.capacity:int = int(capacity)
        self.hist_hi:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_lo:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_color:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_vad:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_vad_ave:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_energy:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_zc:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_var:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )

    def __len__(self):
        return len(self.hist_hi)

    def get_pos(self):
        return self.hist_hi.get_pos()

    def to_pos(self, idx):
        return self.hist_hi.to_pos(idx)

    def to_index(self, pos ):
        return self.hist_hi.to_index(pos)

    def clear(self):
        self.hist_hi.clear()
        self.hist_lo.clear()
        self.hist_color.clear()
        self.hist_vad.clear()
        self.hist_vad_ave.clear()
        self.hist_energy.clear()
        self.hist_zc.clear()
        self.hist_var.clear()

    def to_numpy(self, start:int=None, end:int=None, step:int=None ):
        hi = self.hist_hi.to_numpy(start,end,step)
        lo = self.hist_lo.to_numpy(start,end,step)
        color = self.hist_color.to_numpy(start,end,step)
        vad = self.hist_vad.to_numpy(start,end,step)
        vad_ave = self.hist_vad_ave.to_numpy(start,end,step)
        energy = self.hist_energy.to_numpy(start,end,step)
        zc = self.hist_zc.to_numpy(start,end,step)
        var = self.hist_var.to_numpy(start,end,step)
        return np.vstack( (hi,lo,color,vad,vad_ave,energy,zc,var))

    def to_df(self, start:int=None, end:int=None, step:int=None ):
        df = pd.DataFrame({
            'hi': self.hist_hi.to_numpy(start,end,step),
            'lo': self.hist_lo.to_numpy(start,end,step),
            'color': self.hist_color.to_numpy(start,end,step),
            'vad': self.hist_vad.to_numpy(start,end,step),
            'vad_ave': self.hist_vad_ave.to_numpy(start,end,step),
            'energy': self.hist_energy.to_numpy(start,end,step),
            'zc': self.hist_zc.to_numpy(start,end,step),
            'var': self.hist_var.to_numpy(start,end,step),
        })
        return df

    def add(self, hi, lo, color, vad, energy, zc, var ):
        self.hist_hi.add(hi)
        self.hist_lo.add(lo)
        self.hist_color.add(color)
        self.hist_vad.add(vad)
        self.hist_vad_ave.add(vad)
        self.hist_energy.add(energy)
        self.hist_zc.add(zc)
        self.hist_var.add(var)
        window=self.window
        offset = window//2 - window
        if len(self.hist_vad_ave)+offset>=0:
            rb_frame = self.hist_vad_ave.to_numpy(-window)
            rb_ma = rb_frame.mean()
            self.hist_vad_ave.set(len(self.hist_vad_ave)+offset, rb_ma)
        return self.hist_hi.length

    def get_color( self, idx ):
        self.hist_color.get( idx )

    def set_color( self, idx, color ):
        self.hist_color.set( idx, color )

    def set_var( self, idx, var ):
        self.hist_var.set( idx, var )

    def get_vad_count(self,idx):
        return self.hist_vad.get(idx)

    def get_vad(self,idx):
        try:
            return self.hist_vad.get(idx)
        except:
            return 0.0

    def get_vad_ave(self,idx):
        try:
            return self.hist_vad_ave.get(idx)
        except:
            return 0.0

    def get_vad_slope(self,idx):
        try:
            return self.hist_vad_ave.get(idx) - self.hist_vad_ave.get(idx-1)
        except:
            return 0.0

    def get_vad_accel(self,idx):
        try:
            return self.get_vad_slope(idx) - self.get_vad_slope(idx-1)
        except:
            return 0.0

    # def keep(self,sz):
    #     rm = self.hist_hi.capacity - sz
    #     self.remove(rm)

    # def remove(self, rm ):
    #     self.hist_hi.remove( rm )
    #     self.hist_lo.remove( rm )
    #     self.hist_color.remove( rm )
    #     self.hist_vad.remove( rm )
    #     self.hist_vad_ave.remove( rm )
    #     self.hist_energy.remove( rm )
    #     self.hist_zc.remove( rm )
    #     self.hist_var.remove( rm )

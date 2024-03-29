import numpy as np
import pandas as pd
from .ring_buffer import RingBuffer

class Hists:

    def __init__(self,capacity):
        self.capacity:int = int(capacity)
        self.hist_hi:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_lo:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_color:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_vad_count:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_vad:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
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
        self.hist_vad_count.clear()
        self.hist_vad.clear()
        self.hist_energy.clear()
        self.hist_zc.clear()
        self.hist_var.clear()

    def to_numpy(self, start:int=None, end:int=None, step:int=None ):
        hi = self.hist_hi.to_numpy(start,end,step)
        lo = self.hist_lo.to_numpy(start,end,step)
        color = self.hist_color.to_numpy(start,end,step)
        vad1 = self.hist_vad_count.to_numpy(start,end,step)
        vad2 = self.hist_vad.to_numpy(start,end,step)
        energy = self.hist_energy.to_numpy(start,end,step)
        zc = self.hist_zc.to_numpy(start,end,step)
        var = self.hist_var.to_numpy(start,end,step)
        return np.vstack( (hi,lo,color,vad1,vad2,energy,zc,var))

    def to_df(self, start:int=None, end:int=None, step:int=None ):
        df = pd.DataFrame({
            'hi': self.hist_hi.to_numpy(start,end,step),
            'lo': self.hist_lo.to_numpy(start,end,step),
            'color': self.hist_color.to_numpy(start,end,step),
            'vad1': self.hist_vad_count.to_numpy(start,end,step),
            'vad2': self.hist_vad.to_numpy(start,end,step),
            'energy': self.hist_energy.to_numpy(start,end,step),
            'zc': self.hist_zc.to_numpy(start,end,step),
            'var': self.hist_var.to_numpy(start,end,step),
        })
        return df

    def add(self, hi, lo, color, vad_count, vad, energy, zc, var ):
        self.hist_hi.add(hi)
        self.hist_lo.add(lo)
        self.hist_color.add(color)
        self.hist_vad_count.add(vad_count)
        self.hist_vad.add(vad)
        self.hist_energy.add(energy)
        self.hist_zc.add(zc)
        self.hist_var.add(var)

    def replace_color( self, color ):
        self.hist_color.set(-1,color)

    def replace_var( self, var ):
        self.hist_var.set(-1,var)

    def get_vad_count(self,idx):
        return self.hist_vad_count.get(idx)

    def keep(self,sz):
        rm = self.hist_hi.capacity - sz
        self.remove(rm)

    def remove(self, rm ):
        self.hist_hi.remove( rm )
        self.hist_lo.remove( rm )
        self.hist_color.remove( rm )
        self.hist_vad_count.remove( rm )
        self.hist_vad.remove( rm )
        self.hist_energy.remove( rm )
        self.hist_zc.remove( rm )
        self.hist_var.remove( rm )

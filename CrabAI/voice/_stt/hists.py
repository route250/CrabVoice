import numpy as np
from .ring_buffer import RingBuffer

class Hists:

    def __init__(self,capacity):
        self.capacity:int = int(capacity)
        self.hist_hi:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_lo:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_vad_count:RingBuffer = RingBuffer( self.capacity, dtype=np.int32 )
        self.hist_vad:RingBuffer = RingBuffer( self.capacity, dtype=np.int32 )
        self.hist_energy:RingBuffer = RingBuffer( self.capacity, dtype=np.float32 )
        self.hist_zc:RingBuffer = RingBuffer( self.capacity, dtype=np.int32 )

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
        self.hist_vad_count.clear()
        self.hist_vad.clear()
        self.hist_energy.clear()
        self.hist_zc.clear()

    def to_numpy(self, start:int=None, end:int=None, step:int=None ):
        hi = self.hist_hi.to_numpy(start,end,step)
        lo = self.hist_lo.to_numpy(start,end,step)
        vad1 = self.hist_vad_count.to_numpy(start,end,step)
        vad2 = self.hist_vad.to_numpy(start,end,step)
        energy = self.hist_energy.to_numpy(start,end,step)
        zc = self.hist_zc.to_numpy(start,end,step)
        return np.vstack( (hi,lo,vad1,vad2,energy,zc))

    def add(self, hi, lo, vad_count, vad, energy, zc ):
        self.hist_hi.append( np.array([hi],dtype=np.float32) )
        self.hist_lo.append(  np.array([lo],dtype=np.float32) )
        self.hist_vad_count.append(  np.array([vad_count], dtype=np.int32) )
        self.hist_vad.append(  np.array([vad], dtype=np.int32) )
        self.hist_energy.append(  np.array([energy], dtype=np.float32) )
        self.hist_zc.append(  np.array([zc], dtype=np.int32) )

    def get_vad_count(self,idx):
        return self.hist_vad_count.get(idx)

    def keep(self,sz):
        rm = self.hist_hi.capacity - sz
        self.remove(rm)

    def remove(self, rm ):
        self.hist_hi.remove( rm )
        self.hist_lo.remove( rm )
        self.hist_vad_count.remove( rm )
        self.hist_vad.remove( rm )
        self.hist_energy.remove( rm )
        self.hist_zc.remove( rm )

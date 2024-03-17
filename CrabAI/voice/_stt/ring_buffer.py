from threading import Condition
import numpy as np

class RingBuffer:
    """リングバッファクラス"""
    def __init__(self, capacity:int, *, dtype=np.float32 ):
        """
        コンストラクタ
        capacity: 容量
        dtype: NumPyのdtype
        """
        self._lock:Condition = Condition()
        self.dtype=dtype
        self.capacity:int = int(capacity)
        self.buffer:np.ndarray = np.zeros( self.capacity, dtype=dtype )
        #
        self.offset:int = 0
        self.pos:int = 0
        self.length:int = 0

    def clear(self):
        with self._lock:
            self.offset = 0
            self.pos = 0
            self.length = 0

    def is_full(self) ->bool:
        with self._lock:
            return self.capacity==self.length

    def __len__(self):
        with self._lock:
            return self.length

    def get_pos(self):
        return self.offset+self.length

    def to_pos(self, idx):
        return self.offset+idx

    def to_index(self, pos ):
        return pos - self.offset

    def add(self, value ):
        try:
            item:np.ndarray = np.array( [value] ).astype(self.dtype)
            self.append( item )
        except:
            pass

    def append(self, item: np.ndarray):
        with self._lock:
            item_len = len(item)
            if item_len==0:
                # 追加データの長さがゼロの場合
                return
            if item_len >= self.capacity or self.length==0:
                # 追加データだけで容量を超える場合、または、空状態に追加する場合
                self.offset += max( 0, item_len-self.capacity )
                self.pos = 0
                self.length = min( item_len, self.capacity )
                np.copyto( self.buffer[:self.length], item[-self.length:] )
                return

            copy_start = (self.pos + self.length) % self.capacity  # コピー開始位置
            copy_len = min(item_len, self.capacity - copy_start)   # 折返しまでの長さ
            copy_end = copy_start + copy_len                       # 終了位置
            np.copyto( self.buffer[copy_start:copy_end], item[:copy_len] )
            if copy_len < item_len:
                np.copyto( self.buffer[:item_len-copy_len], item[copy_len:] )

            self.length = self.length + item_len
            if self.length > self.capacity:
                remove_length = self.length - self.capacity
                self.offset += remove_length
                self.pos = (self.pos+remove_length) % self.capacity
                self.length = self.capacity

    def remove(self,length):
        with self._lock:
            if length>=self.length:
                self.offset += self.length
                self.pos=0
                self.length = 0
            else:
                self.offset += length
                self.length -= length
                self.pos = (self.pos+length) % self.capacity

    def to_numpy(self, start=None, end=None, step=None):

        with self._lock:

            # スライスでアクセスされた場合、start, stop, step を正規化
            start0, end0, step0 = slice(start,end,step).indices(self.length)

            # 範囲が存在しない場合
            if start0 >= end0:
                return np.empty(0, dtype=self.dtype)

            # 物理的なインデックスに変換
            start1 = (self.pos + start0) % self.capacity
            end1 = (self.pos + end0) % self.capacity

            # スライスがバッファをまたがない場合
            if start1 < end1:
                return self.buffer[start1:end1:step0].copy()

            # バッファがまたがる場合、2つの部分に分割して結合
            sz1 = self.capacity - start1
            join = np.empty( sz1+end1, dtype=self.dtype)
            if sz1>0:
                # startからバッファの終わりまで
                join[:sz1] = self.buffer[start1:]
            if end1>0:
                # バッファの始まりからstopまで
                join[sz1:] = self.buffer[:end1]
            if step0>1:
                join = join[::step0]
            return join

    def get(self,index:int):
        with self._lock:
            # 単一のインデックスでアクセスされた場合の処理は変更なし
            if index < -self.length or self.length <= index:
                raise IndexError(f"Index {index} out of bounds")
            if 0<=index:
                return self.buffer[ (self.pos + index) % self.capacity ]
            else:
                return self.buffer[ (self.pos + self.length + index) % self.capacity ]

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, slice):
            return self.to_numpy( key.start, key.stop, key.step )
        elif isinstance(key, int):
            return self.get(key)
        else:
            raise TypeError("Invalid argument type.")
                
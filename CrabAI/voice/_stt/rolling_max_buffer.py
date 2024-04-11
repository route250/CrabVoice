import numpy as np

class RollingMaxBuffer:
    """一定区間の最大値を覚えておくクラス"""
    def __init__(self, buffer_length:int ):
        self._buf:np.ndarray = np.empty( buffer_length, dtype=np.float32 )
        self._buf[:] = np.nan
        self._pos:int = 0
        self._max_value:float = np.nan
        self._max_pos:int = 0

    def put(self,value):
        """新しい値を追加する"""
        value = float(value)
        self._buf[self._pos] = value
        if self._max_pos == self._pos:
            # 新規データが最大値を上書きしたので、現在の最大をスキャン
            self._max_pos = np.nanargmax( self._buf )
            self._max_value = float( self._buf[self._max_pos] )
        elif self._max_value < value:
            # 新規データが最大値を超えたので記録する
            self._max_pos = self._pos
            self._max_value = value
        self._pos = (self._pos+1) % len(self._buf)

    def get(self):
        """現在の最大値を取得する"""
        return self._max_value

def test():
    sz = 10
    buf = RollingMaxBuffer(sz)

    data:np.ndarray = ( np.random.rand( 3000 ) * 100 ).astype(np.int32).astype(np.float32)
    # data:np.ndarray = np.array( [ 6,5,4,3,2,1 ])
    testsize = len(data)
    for i in range( testsize ):
        buf.put(data[i])
        s = max(0, i-sz+1)
        e = i+1
        m1 = max( data[s:e] )
        m2 = buf.get()
        if m1 != m2:
            print(f"{i} {s}:{e} {m1} {m2} ERROR")
            break
        elif i<10:
            print(f"{i} {s}:{e} {m1} {m2}")

if __name__ == "__main__":
    test()
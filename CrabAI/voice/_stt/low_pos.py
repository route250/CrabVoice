import numpy as np

class LowPos:
    def __init__(self, max_elements:int=10):
        self.max_elements:int = max_elements
        self.table:np.ndarray = np.full((max_elements, 2), np.inf)
        self.current_size:int = 0
    def __len__(self) ->int:
        return self.current_size
    def clear(self):
        np.ndarray.fill( self.table, np.inf )
        self.current_size = 0

    def push(self, pos: int, value: int):
        # 新しい要素を挿入する位置をバイナリサーチで探す
        insert_at = np.searchsorted(self.table[:self.current_size, 0], value)
        
        # 配列がまだ最大要素数に達していない場合
        if self.current_size < self.max_elements:
            # 挿入位置以降の要素を一つ後ろにシフト
            self.table[insert_at+1:self.current_size+1] = self.table[insert_at:self.current_size]
            self.table[insert_at] = [value, pos]
            self.current_size += 1
        else:
            # 配列が満杯で、挿入位置が配列の末尾よりも前の場合
            if insert_at < self.max_elements - 1:
                # 挿入位置以降の要素を一つ後ろにシフトし、末尾の要素を削除
                self.table[insert_at+1:] = self.table[insert_at:-1]
                self.table[insert_at] = [value, pos]

    def get_table(self) ->np.ndarray:
        return self.table[:self.current_size]

    def get_pos(self):
        return int(self.table[0][1])

    def get_posx(self, start, end ):
        for i in range(0,self.current_size):
            pos = int(self.table[i][1])
            if start<=pos and pos<end:
                for j in range(i+1,self.current_size):
                    self.table[j-1] = self.table[j]
                self.current_size-=1
                return pos
        return -1

    def pop(self):
        if self.current_size<=0:
            return None
        ret = int(self.table[0][1])
        for i in range(1,self.current_size):
            self.table[i-1] = self.table[i]
        self.current_size-=1

    def remove_below_pos(self, pos: int):
        # pos以下の要素を削除する
        new_pos = 0
        for i in range(self.current_size):
            if self.table[i][1]>pos:
                self.table[new_pos] = self.table[i]
                new_pos += 1
                
        # 新しい現在のサイズを更新
        self.current_size = new_pos
        
        # 残りの部分をnp.infで埋める
        if new_pos < self.max_elements:
            self.table[new_pos:] = np.inf
    
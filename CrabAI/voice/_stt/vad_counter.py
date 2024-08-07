import numpy as np
import webrtcvad

class VadTbl:

    def __init__(self,size,up:int,dn:int):
        if up<dn:
            raise Exception(f"invalid parameter {size} {up} {dn}")
        self.size = size
        self.up_trigger:int = up
        self.dn_trigger:int = dn
        self.active:bool = False
        self.table:list[int] = [0] * size
        self.pos:int = 0
        self.sum:int = 0

    def add(self,value:int):
        d: int = value - self.table[self.pos]
        self.sum += d
        self.table[self.pos]=value
        self.pos = ( self.pos + 1 ) % self.size
        if self.active:
            if self.sum<=self.dn_trigger:
                self.active = False
                return True
        else:
            if self.sum>=self.up_trigger:
                self.active = True
                return True
        return False

    def __str__(self):
        return f"value:{self.sum},active:{self.active}"
    def __bool__(self):
        return self.active
    def __int__(self):
        return self.sum
    def __float__(self):
        return float(self.sum)
    def __lt__(self,other):
        if isinstance(other,bool):
            return self.active<other
        else:
            return self.sum<other
    def __le__(self,other):
        if isinstance(other,bool):
            return self.active<=other
        else:
            return self.sum<=other
    def __eq__(self,other):
        if isinstance(other,bool):
            return self.active==other
        else:
            return self.sum==other
    def __ne__(self,other):
        if isinstance(other,bool):
            return self.active!=other
        else:
            return self.sum!=other
    def __gt__(self,other):
        if isinstance(other,bool):
            return self.active>other
        else:
            return self.sum>other
    def __ge__(self,other):
        if isinstance(other,bool):
            return self.active>=other
        else:
            return self.sum>=other

class VadCounter:
    """音声の区切りを検出する"""
    def __init__(self):
        # 設定
        self.fr = 16000
        self.size:int = 10
        self.up_tirg:int = 9
        self.dn_trig:int = 3
        # 判定用 カウンタとフラグ
        self.vad_count = 0
        self.vad_state:bool = False
        # 処理用
        self.hists_tbl:list[bool] = [False] * self.size
        self.hists_pos:int = 0
        self.seg = b''
        self.vad = webrtcvad.Vad()

    def put_f32(self, audio:np.ndarray ) ->tuple[bool,bool,bool,bool]:
        """
        float32の音声データから区切りを検出
        戻り値: start,up,dn,end
        """
        pcm = audio * 32767.0
        pcm = pcm.astype(np.int16)
        return self.put_i16( pcm )

    def put_i16(self, pcm:np.ndarray ) ->tuple[bool,bool,bool,bool]:
        return self.put_bytes( pcm.tobytes() )
    
    def put_bytes(self, data:bytes ) ->tuple[bool,bool,bool,bool]:
        start_state:bool = self.vad_state
        up_trigger:bool = False
        down_trigger:bool = False
        end_state:bool = self.vad_state
        # データ長
        data_len = len(data)
        # 処理単位
        seg_sz = int( (self.fr / 100) * 2 )# 10ms * 2bytes(int16)
        # 前回の居残りデータ
        seg = self.seg
        # 分割範囲初期化
        st=0
        ed = st + seg_sz - len(seg) # 前回の残りを考慮して最初の分割を決める
        # 分割ループ
        while st<data_len:
            # 分割する
            seg += data[st:ed]
            # 処理単位を満たしていれば処理する
            if ed<=data_len:
                if self.vad.is_speech(seg, self.fr):
                    # 有声判定
                    if not self.hists_tbl[self.hists_pos]:
                        self.hists_tbl[self.hists_pos] = True
                        self.vad_count+=1
                else:
                    # 無声判定
                    if self.hists_tbl[self.hists_pos]:
                        self.hists_tbl[self.hists_pos] = False
                        self.vad_count-=1
                self.hists_pos = (self.hists_pos+1) % self.size
                # 居残りクリア
                seg =b''
                # 判定
                if self.vad_state:
                    if self.vad_count<=self.dn_trig:
                        self.vad_state = False
                        down_trigger = True
                else:
                    if self.vad_count>=self.up_tirg:
                        self.vad_state = True
                        up_trigger = True
            st = ed
            ed = st + seg_sz
        self.seg = seg
        end_state:bool = self.vad_state
        return start_state,up_trigger,down_trigger,end_state

import sys
import os
import time
from io import BytesIO
from threading import Thread, Condition
import wave
import traceback
import logging
import numpy as np
import webrtcvad
import librosa
import sounddevice as sd

logger = logging.getLogger('voice')

def _mic_priority(x):
    devid = x['index']
    name = x['name']
    if 'default' in name:
        return 10000 + devid
    if 'USB' in name:
        return 20000 + devid
    return 90000 + devid

def get_mic_devices( *, samplerate=None, dtype=None ):
    """マイクとして使えるデバイスをリストとして返す"""
    # 条件
    sr:float = float(samplerate) if samplerate else 16000
    dtype = dtype if dtype else np.float32
    # select input devices
    inp_dev_list = [ x for x in sd.query_devices() if x['max_input_channels']>0 ]
    # select avaiable devices
    mic_dev_list = []
    for x in inp_dev_list:
        mid = x['index']
        name = f"[{mid:2d}] {x['name']}"
        try:
            # check parameters
            sd.check_input_settings( device=mid, samplerate=sr, dtype=dtype )
            # read audio data
            # channelsを指定したら内臓マイクで録音できないので指定してはいけない。
            with sd.InputStream( samplerate=sr, device=mid ) as audio_in:
                frames,overflow = audio_in.read(1000)
                audio_in.abort(ignore_errors=True)
                audio_in.stop(ignore_errors=True)
                audio_in.close(ignore_errors=True)
                if len(frames.shape)>1:
                    frames = frames[:,0]
                if max(abs(frames))<1e-9:
                    logger.debug(f"NoSignal {name}")
                    continue
            logger.debug(f"Avairable {name}")
            mic_dev_list.append(x)
        except sd.PortAudioError:
            logger.debug(f"NoSupport {name}")
        except:
            logger.exception('mic')
    # sort
    mic_dev_list = sorted( mic_dev_list, key=_mic_priority)
    # for x in mic_dev_list:
    #     print(f"[{x['index']:2d}] {x['name']}")
    return mic_dev_list

def towave( audio, fr = 16000 ):
    if len(audio.shape)==1:
        ch = 1
    elif len(audio.shape)==2:
        ch = audio.shape[1]
    else:
        raise IndexError()
    pcm = audio * 32767.0
    pcm:np.ndarray = pcm.astype(np.int16)
    st = BytesIO()
    with wave.open( st, 'wb') as stream:
        stream.setnchannels(ch)
        stream.setframerate(fr)
        stream.setsampwidth(2)
        stream.writeframes(pcm.tobytes() )
    st.seek(0)
    return st

def load_wave( filename, *, callback, segsize=800, samplerate=16000, wait=True ):
    try:
        librosa.resample( np.zeros( (1,1), dtype=np.float32), orig_sr=samplerate*2, target_sr=samplerate ) # preload of librosa
        with wave.open( filename, 'rb' ) as stream:
            ch = stream.getnchannels()
            fr = stream.getframerate()
            sw = stream.getsampwidth()
            num_fr = stream.getnframes()
            num_tm = num_fr/fr
            wait_time = segsize/fr
            log_inverval = fr * 5
            i = 0
            l = 0
            start_time = time.time()
            frame_readed = 0
            while True:
                x = stream.readframes(segsize)
                if x is None or len(x)==0:
                    break
                i+=segsize
                pcm = np.frombuffer( x, dtype=np.int16).reshape(-1,ch)
                frame_readed += len(pcm)
                call_time = start_time + (frame_readed/fr)
                l += len(pcm)
                if l>log_inverval:
                    l=0
                    tm = i/fr
                    print( f"wave {tm:.2f}/{num_tm:.2f} {i}/{num_fr}")
                orig_audio_f32 = pcm /32767.0
                if fr != samplerate:
                    audio_f32 = librosa.resample( orig_audio_f32, axis=0, orig_sr=fr, target_sr=samplerate )
                else:
                    audio_f32 = orig_audio_f32
                wa = call_time - time.time()
                if wait and wa>0:
                    time.sleep( wa )
                #else:
                    #print( "call slow!! STOP!!" )
                    #break
                callback( audio_f32 )
    except:
        traceback.print_exc()

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
        self.hists_tbl:list[bool] = [0] * self.size
        self.hists_pos:int = 0
        self.seg = b''
        self.vad = webrtcvad.Vad()

    def put_f32(self, audio:np.ndarray ) ->bool:
        """
        float32の音声データから区切りを検出
        戻り値: start,up,dn,end
        """
        pcm = audio * 32767.0
        pcm = pcm.astype(np.int16)
        return self.put_i16( pcm )

    def put_i16(self, pcm:np.ndarray ) ->bool:
        return self.put_bytes( pcm.tobytes() )
    
    def put_bytes(self, data:bytes ) ->list[bool]:
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
                
class AudioRingBuffer(RingBuffer):

    def __init__(self, fr:int=16000, sec:float=30):
        self.fr:int = fr if isinstance(fr,int) else 16000
        self.sec:float = sec if isinstance(sec,float) else 30.0
        super().__init__( int(self.fr * self.sec), dtype=np.float32 )

    def towave(self) ->BytesIO:
        wav = towave( self[0:], fr=self.fr )
        return wav

    def get_last_fr(self):
        return self.offset + self.length

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

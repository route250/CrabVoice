
import time
import numpy as np
from voice_utils import RingBuffer

class RingTest:
    def __init__(self,size:int,dtype):
        self.size:int = size
        self.buffer:np.ndarray = np.zeros(0,dtype=dtype)

    def __len__(self):
        return len(self.buffer)

    def append(self, item):
        data = np.concatenate( (self.buffer,item), axis=0 )
        n = min( len(data), self.size )
        self.buffer = data[-n:]

    def remove(self, n ):
        self.buffer = self.buffer[n:]

    def __getitem__(self, key) -> np.ndarray:
        return self.buffer[key]

def eq( a,b ):
    if isinstance(a,int) and isinstance(b,int):
        return a==b
    if isinstance(a,str) and isinstance(b,str):
        return a==b
    if isinstance(a,float) and isinstance(b,float):
        return a==b
    if isinstance(a,np.int16) and isinstance(b,np.int16):
        return a==b
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        return np.array_equiv(a,b)
    return False

def ringbuffer_test():
    EXMSG="exception"

    No = 0

    for buffer_size in range(1,12,3):
        for append_size in range(0,buffer_size+2):
            pos_list = sorted(set([ 0, 1, buffer_size//2, buffer_size-1,buffer_size ]))
            for pos in pos_list:
                for remove_size in range(0,buffer_size+2):
                    No += 1
                    TestCase=f"テスト{No} sz:{buffer_size} add:{append_size} del:{remove_size}"
                    Base:RingTest = RingTest( buffer_size,dtype=np.int16)
                    Ring:RingBuffer = RingBuffer( buffer_size,dtype=np.int16)
                    if pos>0:
                        data0:np.ndarray = np.array( [x for x in range(1001,1001+pos)], dtype=np.int16 )
                        Base.append(data0)
                        Ring.append(data0)
                        if pos>2:
                            Base.remove(pos-1)
                            Ring.remove(pos-1)

                    data1:np.ndarray = np.array( [x for x in range(1,append_size+1)], dtype=np.int16 )
                    Base.append(data1)
                    Ring.append(data1)

                    for idx in range(len(data1)):
                        data1[idx]=-1

                    if remove_size>0:
                        Base.remove( remove_size )
                        Ring.remove( remove_size )

                    act_len=len(Base)
                    exp_len=len(Ring)
                    if act_len != exp_len:
                        print( f"{TestCase} len()が一致しない")

                    try:
                        act = Base[:]
                    except Exception as ex:
                        act = EXMSG # str(ex)
                    try:
                        exp = Ring[:]
                    except Exception as ex:
                        exp = EXMSG # str(ex)
                    if not eq(act,exp):
                        print( f"{TestCase} [:] が一致しない {act} vs {exp}")

                    for idx in range(-act_len-1,act_len+1):
                        try:
                            act = Base[idx]
                        except Exception as ex:
                            act = EXMSG # str(ex)
                        try:
                            exp = Ring[idx]
                        except Exception as ex:
                            exp = EXMSG # str(ex)
                        if not eq(act,exp):
                            print( f"{TestCase} [{idx}] が一致しない {act} vs {exp}")

                    for idx in range(-act_len-1,act_len+1):
                        try:
                            act = Base[idx:]
                        except Exception as ex:
                            act = EXMSG # str(ex)
                        try:
                            exp = Ring[idx:]
                        except Exception as ex:
                            exp = EXMSG # str(ex)
                        if not eq(act,exp):
                            print( f"{TestCase} [{idx}:] が一致しない {act} vs {exp}")

                    for idx in range(-act_len-1,act_len+1):
                        try:
                            act = Base[:idx]
                        except Exception as ex:
                            act = EXMSG # str(ex)
                        try:
                            exp = Ring[:idx]
                        except Exception as ex:
                            exp = EXMSG # str(ex)
                        if not eq(act,exp):
                            print( f"{TestCase} [:{idx}] が一致しない {act} vs {exp}")

                    for begin in range(-act_len-1,act_len+1):
                        for end in range(-act_len-1,act_len+1):
                            try:
                                act = Base[begin:end]
                            except Exception as ex:
                                act = EXMSG # str(ex)
                            try:
                                exp = Ring[begin:end]
                            except Exception as ex:
                                exp = EXMSG # str(ex)
                            if not eq(act,exp):
                                print( f"{TestCase} [{begin}:{end}] が一致しない {act} vs {exp}")
                                act = Base[begin:end]
                                exp = Ring[begin:end]
def perf_test():

    Ring:RingBuffer = RingBuffer(16000*30,dtype=np.float32)

    data1 = np.random.uniform( low=-1,high=1,size=800 )

    Ring.append(data1)
    print( f"Perf: len:{len(Ring)}")

    t0 = time.time()
    n=100000
    i=0
    while i<n:
        Ring.append(data1)
        i += 1
    t1 = time.time()
    t = t1-t0
    dt = t/n
    print( f"Perf: len:{len(Ring)}")
    print( f" {t:.4f}(sec) {dt:.6f}(msec)")


def main():
    data0 = [x for x in range(1,6)]
    print( f"test: {data0[1:1]}")
    print( f"test: {data0[1:2]}")
    print( f"test: {data0[2:4]}")

    Sp = RingBuffer(10,dtype=np.int16)
    print( f"len:{len(Sp)}")
    print( f"dump {Sp[:]}")
    
    data1:np.ndarray = np.array( [x for x in range(1,4)], dtype=np.int16 )
    Sp.append(data1)
    print( f"append {data1[:]} dump {Sp[:]}")

    data2:np.ndarray = np.array( [x for x in range(11,14)], dtype=np.int16 )
    Sp.append(data2)
    print( f"append {data2[:]} dump {Sp[:]}")

    data3:np.ndarray = np.array( [x for x in range(101,104)], dtype=np.int16 )
    Sp.append(data3)
    print( f"append {data3[:]} dump {Sp[:]}")

    data4:np.ndarray = np.array( [x for x in range(21,24)], dtype=np.int16 )
    Sp.append(data4)
    print( f"append {data4[:]} dump {Sp[:]}")

    for i in range(len(Sp)):
        j = i - len(Sp)
        print( f"i:{i:3d} {Sp[i]:3d} {Sp[i:i+1]} i:{j:3d} {Sp[j]:3d} {Sp[j:j+1]}")

    Sp.remove(1)
    print( f"remove 1 {len(Sp)} dump {Sp[:]}")

    Sp.remove(3)
    print( f"remove 1 {len(Sp)} dump {Sp[:]}")

if __name__ == "__main__":
    ringbuffer_test()
    perf_test()
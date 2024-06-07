import sys,os
import time
from logging import getLogger
import traceback
import multiprocessing as mp
from multiprocessing import Process, Array
from multiprocessing.queues import Queue as PQ
from heapq import heapify, heappop, heappush
from queue import Empty
import copy
import numpy as np

class Ev:
    Nop:int = 0
    Config:int = 10
    Load:int = 100
    Start:int = 101
    Stop:int = 102
    StartOfData:int=1000
    EndOfData:int=1001

    def __init__(self, seq:int, typ, *args, **kwargs ):
        self.seq = seq
        self.proc_no=None
        self.num_proc=None
        self.typ = typ
        self.args = args
        self.kwargs = kwargs

    def set_seq(self,seq:int):
        if isinstance(seq,int):
            self.seq = seq
        else:
            raise ValueError("invalid seq {seq}")

    @staticmethod
    def type_to_str(typ:int):
        if Ev.Nop==typ:
            return "Nop"
        elif Ev.Config==typ:
            return "Config"
        elif Ev.Load==typ:
            return "Load"
        elif Ev.Start==typ:
            return "Start"
        elif Ev.Stop==typ:
            return "Stop"
        elif Ev.StartOfData==typ:
            return "StartOfData"
        elif Ev.EndOfData==typ:
            return "EndOfData"

    def is_distribution(self):
        if Ev.Config==self.typ or Ev.StartOfData==self.typ or Ev.EndOfData==self.typ:
            return True
        else:
            return False

    def is_collection(self):
        if Ev.StartOfData==self.typ or Ev.EndOfData==self.typ:
            return True
        else:
            return False

    def __str__(self) ->str:
        no = self.proc_no if isinstance(self.proc_no,int) else ''
        num = f"/{self.num_proc}" if isinstance(self.num_proc,int) else ''
        seq = f"#{self.seq}, " if isinstance(self.seq,int) else ''
        return f"[{no}{num}{seq}{Ev.type_to_str(self.typ)}, {self.args} {self.kwargs} ]"

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f'Key must be a string, not {type(key).__name__}')
        if key == 'seq':
            return self.seq if self.seq is not None else 0
        if key == 'typ':
            return self.typ if self.typ is not None else None
        raise KeyError(f'Key {key} not found')

IDX_SEQ = 0
IDX_STAT_MAIN = 1

IDX_VAD_PICK = 10
IDX_VAD_UP = 11
IDX_VAD_DN = 12
IDX_VAD_IGNORE_SEC = 13
IDX_VAD_MIN_SEC = 14
IDX_VAD_MAX_SEC = 15
IDX_VAD_POST_SEC = 16
IDX_VAD_SILENT_SEC = 17
IDX_VAD_VAR = 18

IDX_VOICE_VAR = 20
IDX_VOICE_MAX_SEC = 21
IDX_VAR2 = 61

IDX_AUX = 70
IDX_BUTTER1 = 80
IDX_BUTTER2 = 90

class ShareParam:

    def __init__(self, share ):
        self._share_array = share
        self._share_key = share[0]

    def _set_value(self, idx, value, *, notify=True ) ->float:
        if isinstance(value,(float,int)):
            value = float(value)
            if self._share_array[idx] != value:
                self._share_array[idx] = value
                if notify:
                    self._share_array[0] = self._share_array[0] + 1.0
            return value
        return None

    def _set_list(self, idx, sz, values, *, notify=True ):
        if not isinstance(values,list) or len(values)!=sz:
            return None
        for v in values:
            if not isinstance(v,(float,int)):
                return None
        update = False
        for i,v in enumerate(values):
            v = float(v)
            if v != self._share_array[idx+i]:
                self._share_array[idx+i] = v
                update = True
        if notify and update:
            self._share_array[0] = self._share_array[0] + 1.0
        return tuple( v for v in self._share_array[idx:idx+sz])

    def _get_value(self, idx ):
        self._share_key = self._share_array[0]
        return self._share_array[idx]

    def _get_list(self, idx, sz ):
        self._share_key = self._share_array[0]
        return tuple( v for v in self._share_array[idx:idx+sz])

    def is_updated(self):
        return self._share_key != self._share_array[0]
    
    def set_stat_main(self, value ):
        return self._set_value( IDX_STAT_MAIN, value )
    def get_stat_main(self ):
        return self._get_value( IDX_STAT_MAIN )

    def set_vad_pick(self, value, *, notify=True ):
        return self._set_value( IDX_VAD_PICK, value, notify=notify )
    def get_vad_pick(self ):
        return self._get_value( IDX_VAD_PICK )

    def set_vad_up(self, value, *, notify=True ):
        return self._set_value( IDX_VAD_UP, value, notify=notify )
    def get_vad_up(self ):
        return self._get_value( IDX_VAD_UP )

    def set_vad_dn(self, value, *, notify=True ):
        return self._set_value( IDX_VAD_DN, value, notify=notify )
    def get_vad_dn(self ):
        return self._get_value( IDX_VAD_DN )

    def set_vad_ignore_sec(self, value, *, notify=True  ):
        return self._set_value( IDX_VAD_IGNORE_SEC, value, notify=notify )
    def get_vad_ignore_sec(self ):
        return self._get_value( IDX_VAD_IGNORE_SEC )

    def set_vad_min_sec(self, value, *, notify=True  ):
        return self._set_value( IDX_VAD_MIN_SEC, value, notify=notify )
    def get_vad_min_sec(self ):
        return self._get_value( IDX_VAD_MIN_SEC )

    def set_vad_max_sec(self, value, *, notify=True  ):
        return self._set_value( IDX_VAD_MAX_SEC, value, notify=notify )
    def get_vad_max_sec(self ):
        return self._get_value( IDX_VAD_MAX_SEC )

    def set_vad_post_sec(self, value, *, notify=True  ):
        return self._set_value( IDX_VAD_POST_SEC, value, notify=notify )
    def get_vad_post_sec(self ):
        return self._get_value( IDX_VAD_POST_SEC )

    def set_vad_silent_sec(self, value, *, notify=True  ):
        return self._set_value( IDX_VAD_SILENT_SEC, value, notify=notify )
    def get_vad_silent_sec(self ):
        return self._get_value( IDX_VAD_SILENT_SEC )

    def set_vad_var(self, value, *, notify=True  ):
        return self._set_value( IDX_VAD_VAR, value, notify=notify )
    def get_vad_var(self ):
        return self._get_value( IDX_VAD_VAR)

    def set_voice_var(self, value, *, notify=True  ):
        return self._set_value( IDX_VOICE_VAR, value, notify=notify )
    def get_voice_var(self ):
        return self._get_value( IDX_VOICE_VAR)

    def set_voice_max_sec(self, value, *, notify=True  ):
        return self._set_value( IDX_VOICE_MAX_SEC, value, notify=notify )
    def get_voice_max_sec(self ):
        return self._get_value( IDX_VOICE_MAX_SEC)

    def set_butter1(self, params, *, notify=True ):
        return self._set_list( IDX_BUTTER1, 4, params, notify=notify)
    def get_butter1(self):
        return self._get_list( IDX_BUTTER1,4)

    def set_butter2(self, params, *, notify=True  ):
        return self._set_list( IDX_BUTTER2, 4, params, notify=notify)
    def get_butter2(self):
        return self._get_list( IDX_BUTTER2,4)

    def set_aux(self, color, vad, en, zc, mute):
        return self._set_list( IDX_AUX, 5, (color,vad,en,zc,mute), notify=False)
    def get_aux(self):
        return self._get_list( IDX_AUX,5)

class VFunction:

    def __init__(self, proc_no:int, num_proc:int, share, data_in:PQ, data_out:PQ ):
        self.conf:ShareParam = ShareParam(share)
        self.num_proc:int = num_proc if isinstance(num_proc,int) and num_proc>0 else 1
        self.proc_no:int = proc_no if isinstance(proc_no,int) and 0<=proc_no and proc_no<self.num_proc else 0
        self.proc_name:str = f"{self.__class__.__name__}#{self.proc_no}"
        self._logger = getLogger(self.__class__.__name__)
        self.enable_in = True
        self.data_in:PQ = data_in
        self.data_out:PQ = data_out
        self.seq_count:int = 0
        self.req_brake:bool = False
        #
        self.input_last_seq:int = 0
        self.input_queue:list = []
        heapify(self.input_queue)

    def debug(self,msg,*args,**kwargs):
        print(f"[{self.proc_name}] {msg}")
        self._logger.debug( f"[{self.proc_name}]{msg}",*args,**kwargs)

    def info(self,*args,**kwargs):
        self._logger.info(*args,**kwargs)

    def error(self,*args,**kwargs):
        self._logger.error(*args,**kwargs)

    def reload_share_param(self):
        return

    def be_distribution(self, ev:Ev ):
        if ev is None or self.num_proc<=1:
            return None # シングルプロセスならNone
        if not ev.is_distribution():
            return None # 配布する必要がなければ None
        # 確認
        new_kwargs = ev.kwargs if isinstance(ev.kwargs,dict) else {}
        dist_list = new_kwargs.get('dist')
        if isinstance(dist_list,(list,tuple)):
            if self.proc_no not in dist_list:
                dist_list = tuple(dist_list)+(self.proc_no,)
        else:
            dist_list =(self.proc_no,)
        new_kwargs['dist']=dist_list
        ev.kwargs=new_kwargs
        if len(dist_list)==self.num_proc:
            # 配布済みなのでNone
            return None
        # コピーを作る
        new_ev:Ev = copy.deepcopy(ev)
        return new_ev

    def is_collected(self,ev:Ev):
        if ev is None:
            return False
        if self.num_proc<=1 or not ev.is_collection():
            return True
        # 確認
        dist_list = ev.kwargs.get('dist') if isinstance(ev.kwargs,dict) else None
        num = len(dist_list) if isinstance(dist_list,(list,tuple)) else 0
        if num==self.num_proc:
            return True
        else:
            return False

    def _get_next_data(self):

        dbg:bool = False # self.__class__.__name__ != "AudioToSegment"

        if self.num_proc<=1:
            # キューを処理する
            if len(self.input_queue)>0 and self.input_queue[0][0]==self.input_last_seq+1:
                seq, ev = heappop(self.input_queue)
                self.input_last_seq=seq
                if ev is not None:
                    if dbg:
                        print(f"[{self.proc_name}] IN Q heappop {str(ev)}")
                    return ev

        if self.num_proc>1:
            try:
                ev:Ev = self.data_in.get( timeout=0.5 )
                if ev is None:
                    return None
                new_ev:Ev = self.be_distribution( ev )
                if new_ev is not None:
                    print(f"[{self.proc_name}] Broadcast {str(new_ev)}")
                    self.data_in.put(new_ev)
                if ev.seq>0:
                    self.input_last_seq = ev.seq
                return ev
            except Empty:
                return None

        while True:

            # get next data
            try:
                ev:Ev = self.data_in.get( timeout=0.5 )
            except Empty:
                return None
            # pass through if seq<=0
            if ev.seq<=0:
                if dbg:
                    print(f"[{self.proc_name}] IN Q pass through {str(ev)}")
                return ev

            if self.input_last_seq+1==ev.seq:
                # 順番が一致していればそのままコール
                self.input_last_seq=ev.seq
                if dbg:
                    print(f"[{self.proc_name}] IN Q seq  {str(ev)}")
                return ev
            else:
                # 順番が来てなければキューに入れる
                if dbg:
                    print(f"[{self.proc_name}] IN Q heappush {str(ev)}")
                heappush( self.input_queue, (ev.seq,ev) )

    def _event_loop(self):

        loop_count:int = 0
        while not self.req_brake:
            loop_count+=1
            try:
                ev:Ev = self._get_next_data()
                if ev is None:
                    if self.conf.is_updated():
                        if self.conf.get_stat_main() != 0.0:
                            self.req_break = True
                        else:
                            self.reload_share_param()
                    continue

                # self.debug(f"Ev {ev}")
                if not isinstance(ev.typ,int):
                    ev.typ = Ev.Nop
                try:
                    if ev.typ == Ev.Config:
                        self.reload_share_param()
                    else:
                        self.proc(ev)
                except:
                    traceback.print_exc()

                try:
                    if ev.typ == Ev.EndOfData or ev.typ == Ev.Stop:
                        self.proc_output_event(ev)
                        break
                except:
                    traceback.print_exc()
                    break

            except:
                traceback.print_exc()
                break
        self.stop()
        print(f"[{self.proc_name}] End")

    def _event_start(self):

        print(f"[{self.proc_name}] Load")
        self.load()

        if self.enable_in:
            print(f"[{self.proc_name}] Event Start")
            self._event_loop()
        else:
            print(f"[{self.proc_name}] Process Start")
            self.proc(None)
            ev = Ev(self.seq_count, Ev.EndOfData )
            self.proc_output_event(ev)
            self.stop()

    def dbg_output(self, cmd, *args, **kwargs):
        ev = Ev(self.seq_count, cmd, *args, **kwargs )
        self.proc_output_event(ev)

    def proc_output_event(self, ev:Ev):
        if isinstance(self.data_out,PQ):
            if ev.typ == Ev.EndOfData or ev.typ == Ev.Stop:
                if self.is_collected(ev):
                    if self.num_proc>1:
                        print(f"[{self.proc_name}] Q OUT {str(ev)}")
                else:
                    return
            if self.num_proc<=1:
                ev.seq = self.seq_count
                self.seq_count += 1
            if isinstance(ev.kwargs,dict) and 'dist' in ev.kwargs:
                del ev.kwargs['dist']
            ev.proc_no=self.proc_no
            ev.num_proc=self.num_proc
            self.data_out.put(ev)

    def load(self ):
        raise NotImplementedError()

    def proc(self, ev:Ev ):
        raise NotImplementedError()

    def proc_end_of_data(self, ev:Ev ):
        raise NotImplementedError()

    def stop(self):
        return

class VProcess(Process):

    def _dummy():
        pass

    def __init__(self, cls:type[VFunction], proc_no:int, num_proc:int, share, data_in:PQ, data_out:PQ, *args, **kwargs ):
        self.share = share
        self.num_proc:int = num_proc if isinstance(num_proc,int) and num_proc>0 else 1
        self.proc_no:int = proc_no if isinstance(proc_no,int) and 0<=proc_no and proc_no<self.num_proc else 0
        self.proc_name:str = f"{self.__class__.__name__}#{self.proc_no}"
        super().__init__( name=f"{self.proc_name}",target=VProcess._dummy, daemon=True )
        self.cls:type[VFunction] = cls
        self.proc_no:int = proc_no
        self.num_proc:int = num_proc
        self.data_in:PQ = data_in
        self.data_out:PQ = data_out
        self.args = args
        self.kwargs = kwargs
        self.__config_seq:int = 0

    def run(self):
        instance = self.cls( self.proc_no, self.num_proc, self.share, self.data_in, self.data_out, *self.args, **self.kwargs )
        instance._event_start()

class VProcessGrp:

    def __init__(self, cls:type[VFunction], num_proc:int, share, data_in:PQ, data_out:PQ, *args, **kwargs ):
        self.share = share
        self.cls:type[VFunction] = cls
        self.proc_list:list[VProcess] = [ VProcess( cls, i, num_proc, self.share, data_in, data_out, *args, **kwargs) for i in range(num_proc) ]

    def start(self):
        for proc in self.proc_list:
            proc.start()

    def is_alive(self):
        for proc in self.proc_list:
            if proc.is_alive():
                return True
        return False

    def join(self):
        for proc in self.proc_list:
            proc.join()

class PipeA(VFunction):

    def load(self):
        pass

    def reload_share_param(self):
        value = self.conf.get_vad_up()
        self.debug( f"reload vad_up={value}")

    def proc(self, ev ):
        self.debug( f"{str(ev)}")

class PipeB(VFunction):

    def load(self):
        pass

    def reload_share_param(self):
        value = self.conf.get_vad_dn()
        self.debug( f"reload vad_dn={value}")

    def proc(self, ev ):
        self.debug( f"{str(ev)}")

class PipeC(VFunction):

    def load(self):
        pass

    def reload_share_param(self):
        value = self.conf.get_vad_min_sec()
        self.debug( f"reload vad_min_sec={value}")

    def proc(self, ev ):
        self.debug( f"{str(ev)}")

def main():

    share = Array( 'd', 256 )
    conf:ShareParam = ShareParam(share)

    q1:PQ = PQ()
    q2:PQ = PQ()
    q3:PQ = PQ()
    q4:PQ = PQ()

    pa = VProcess( PipeA, share, q1, q2 )
    pb = VProcess( PipeB, share, q2, q3 )
    pc = VProcess( PipeC, share, q3, q4 )

    print(f"Start")
    pa.start()
    pb.start()
    pc.start()

    sr:int = 16000
    seglen:int = int(sr*0.1)
    duration:int = sr* 300
    total=duration//seglen
    ts=time.time()
    for seq in range(total):
        #seg:np.ndarray = np.random.rand( seglen, dtype=np.float32 )
        seg:np.ndarray = np.full( seglen, float(seq+1), dtype=np.float32 )
        ev = Ev(seq, Ev.Audio, seg )
        q1.put(ev)
    ev = Ev(seq, 0 )
    q1.put(ev)
    
    nn = 0
    while nn<=total:
        try:
            ev = q4.get(timeout=0.1)
            # print( f"q4 {ev.seq} {ev.cmd}")
            nn+=1
        except:
            continue

    print(f"pa.join")
    pa.join()
    print(f"pb.join")
    pb.join()
    print(f"pc.join")
    pc.join()
    te = time.time()
    elaps = te-ts
    print( f"Time: {elaps:.2f}")

def config_test():

    share = Array( 'd', 256 )
    conf:ShareParam = ShareParam(share)

    q1:PQ = mp.Queue()
    q2:PQ = mp.Queue()
    q3:PQ = mp.Queue()

    pa = VProcessGrp( PipeA, 1, share, q1, q2 )
    pb = VProcessGrp( PipeB, 1, share, q2, q3 )

    print(f"Start")
    pa.start()
    pb.start()
    time.sleep(3.0)

    print("### set_vad_up(0.1)")
    conf.set_vad_up( 0.1 )
    time.sleep(1.0)
    print("### set_vad_dn(0.1)")
    conf.set_vad_dn( 0.1 )
    time.sleep(1.0)
    print("### set_vad_up(0.2)")
    conf.set_vad_up( 0.2 )
    print("### set_vad_dn(0.2)")
    conf.set_vad_dn( 0.2 )
    time.sleep(1.0)

    seq = 0
    ev = Ev(seq, Ev.EndOfData )
    q1.put(ev)
    print(f"pa.join")
    pa.join()
    print(f"pb.join")
    pb.join()

if __name__ == "__main__":
    config_test()
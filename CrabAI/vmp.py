import sys,os
import time
from logging import getLogger
import traceback
from multiprocessing import Process
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

class VFunction:

    def __init__(self, proc_no:int, num_proc:int, data_in:PQ, data_out:PQ, ctl_out:PQ ):
        self.num_proc:int = num_proc if isinstance(num_proc,int) and num_proc>0 else 1
        self.proc_no:int = proc_no if isinstance(proc_no,int) and 0<=proc_no and proc_no<self.num_proc else 0
        self.proc_name:str = f"{self.__class__.__name__}#{self.proc_no}"
        self._logger = getLogger(self.__class__.__name__)
        self.enable_in = True
        self.data_in:PQ = data_in
        self.data_out:PQ = data_out
        self.ctl_out:PQ = ctl_out
        self.seq_count:int = 0
        self.req_brake:bool = False
        #
        self.input_last_seq:int = 0
        self.input_queue:list = []
        heapify(self.input_queue)

    def debug(self,msg,*args,**kwargs):
        self._logger.debug( f"[{self.proc_name}]{msg}",*args,**kwargs)

    def info(self,*args,**kwargs):
        self._logger.info(*args,**kwargs)

    def error(self,*args,**kwargs):
        self._logger.error(*args,**kwargs)

    def _event_configure(self,ev:Ev):
        if isinstance(ev.kwargs,dict):
            for key,val in ev.kwargs.items():
                self.configure(key,val)
        if isinstance(self.ctl_out,PQ):
            ret = self.to_dict()
            if not isinstance(ret,dict):
                ret = {}
            self.ctl_out.put( Ev( ev.seq, ev.typ, **ret) )

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
                    continue

                self.debug(f"Ev {ev}")
                if not isinstance(ev.typ,int):
                    ev.typ = Ev.Nop
                try:
                    if ev.typ == Ev.Config:
                        self._event_configure(ev)
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

    def output_ctl(self, ev:Ev):
        if isinstance(self.ctl_out,PQ):
            ev.proc_no=self.proc_no
            ev.num_proc=self.num_proc
            self.ctl_out.put(ev)

    def configure(self,key,val):
        return

    def to_dict(self):
        return {}

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

    def __init__(self, cls:type[VFunction], proc_no:int, num_proc:int, data_in:PQ, data_out:PQ, ctl_out:PQ, *args, **kwargs ):
        self.num_proc:int = num_proc if isinstance(num_proc,int) and num_proc>0 else 1
        self.proc_no:int = proc_no if isinstance(proc_no,int) and 0<=proc_no and proc_no<self.num_proc else 0
        self.proc_name:str = f"{self.__class__.__name__}#{self.proc_no}"
        super().__init__( name=f"{self.proc_name}",target=VProcess._dummy, daemon=True )
        self.cls:type[VFunction] = cls
        self.proc_no:int = proc_no
        self.num_proc:int = num_proc
        self.data_in:PQ = data_in
        self.data_out:PQ = data_out
        self.ctl_out:PQ = ctl_out
        self.args = args
        self.kwargs = kwargs
        self.__config_seq:int = 0

    def run(self):
        instance = self.cls( self.proc_no, self.num_proc, self.data_in, self.data_out, self.ctl_out,  *self.args, **self.kwargs )
        instance._event_start()

    def _configure(self, **kwargs ):
        ev = Ev(  self.__config_seq, Ev.Config, **kwargs )
        self.data_in.put( ev )

class VProcessGrp:

    def __init__(self, cls:type[VFunction], num_proc:int, data_in:PQ, data_out:PQ, ctl_out:PQ, *args, **kwargs ):

        self.proc_list:list = [ VProcess( cls, i, num_proc, data_in, data_out, ctl_out, *args, **kwargs) for i in range(num_proc) ]

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

    def proc(self, ev ):
        if ev.cmd == Ev.Audio and len(ev.args)>0:
            audio:np.ndarray = ev.args[0]
            # print(f"[{self.__class__.__name__}] {ev.seq} {ev.cmd} {audio.shape} {audio.dtype}")
            audio2 = audio + 0.1
            self.dbg_output( Ev.Audio, audio2 )

class PipeB(VFunction):
    def proc(self, ev ):
        if ev.cmd == Ev.Audio and len(ev.args)>0:
            audio:np.ndarray = ev.args[0]
            # print(f"[{self.__class__.__name__}] {ev.seq} {ev.cmd} {audio.shape} {audio.dtype}")
            audio2 = audio + 0.1
            self.dbg_output( Ev.Audio, audio2 )

class PipeC(VFunction):
    def proc(self, ev ):
        if ev.cmd == Ev.Audio and len(ev.args)>0:
            audio:np.ndarray = ev.args[0]
            # print(f"[{self.__class__.__name__}] {ev.seq} {ev.cmd} {audio.shape} {audio.dtype}")
            audio2 = audio + 0.1
            self.dbg_output( Ev.Audio, audio2 )

def main():

    c1:PQ = PQ()

    q1:PQ = PQ()
    q2:PQ = PQ()
    q3:PQ = PQ()
    q4:PQ = PQ()

    pa = VProcess( PipeA, c1, q1, q2 )
    pb = VProcess( PipeB, c1, q2, q3 )
    pc = VProcess( PipeC, c1, q3, q4 )

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

if __name__ == "__main__":
    main()
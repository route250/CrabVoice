import sys,os
import time
from logging import getLogger
import traceback
from multiprocessing import Process
from multiprocessing.queues import Queue
from heapq import heapify, heappop, heappush
from queue import Empty
import numpy as np

class Ev:
    Start:int = 11
    EndOfData:int=100
    Stop:int = 0
    Config:int = 3
    Nop:int = 1

    def __init__(self, seq:int, typ, *args, **kwargs ):
        self.seq = seq
        self.no=None
        self.pmax=None
        self.typ = typ
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def type_to_str(typ:int):
        if Ev.Config==typ:
            return "Config"
        if Ev.Start==typ:
            return "Start"
        if Ev.Stop==typ:
            return "Stop"
        if Ev.EndOfData==typ:
            return "EndOfData"
        if Ev.Nop==typ:
            return "Nop"

    def __str__(self) ->str:
        no = self.no if isinstance(self.no,int) else ''
        pmax = f"/{self.pmax}" if isinstance(self.pmax,int) else ''
        return f"[{no}{pmax} #{self.seq}, {Ev.type_to_str(self.typ)}, {self.args} {self.kwargs} ]"

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f'Key must be a string, not {type(key).__name__}')
        if key == 'seq':
            return self.seq if self.seq is not None else 0
        if key == 'typ':
            return self.typ if self.typ is not None else None
        raise KeyError(f'Key {key} not found')

class VFunction:

    def __init__(self, data_in:Queue, data_out:Queue, ctl_out:Queue, *, no:int=0, pmax:int=0):
        self.no:int = no if isinstance(no,int) and no>0 else 0
        self.pmax:int = pmax if self.no>0 else 0
        self.title:str = f"{self.__class__.__name__}:{self.no}"
        self._logger = getLogger(self.__class__.__name__)
        self.enable_in = True
        self.data_in:Queue = data_in
        self.data_out:Queue = data_out
        self.ctl_out:Queue = ctl_out
        self.seq_count:int = 0
        self.req_brake:bool = False
        #
        self.output_next_seq:int = 0
        self.output_queue:list = []
        heapify(self.output_queue)

    def debug(self,msg,*args,**kwargs):
        self._logger.debug( f"[{self.title}]{msg}",*args,**kwargs)

    def info(self,*args,**kwargs):
        self._logger.info(*args,**kwargs)

    def error(self,*args,**kwargs):
        self._logger.error(*args,**kwargs)

    def _event_configure(self,ev:Ev):
        if isinstance(ev.kwargs,dict):
            for key,val in ev.kwargs.items():
                self.configure(key,val)
        if isinstance(self.ctl_out,Queue):
            ret = self.to_dict()
            if not isinstance(ret,dict):
                ret = {}
            self.ctl_out.put( Ev( ev.seq, ev.typ, **ret) )

    def _get_next_data(self):

        dbg:bool = False # self.__class__.__name__ != "AudioToSegment"
        if self.no>0:
            try:
                while True:
                    ev:Ev = self.data_in.get( timeout=0.5 )
                    if ev.typ==Ev.Config or ev.typ==Ev.EndOfData:
                        no = ev.no if isinstance(ev.no,int) and ev.no>0 else 0
                        if no==0:
                            # プロセス全部にブロードキャスト
                            for i in range(1,self.pmax+1):
                                ev2:Ev = Ev(ev.seq, ev.typ, **ev.kwargs )
                                ev2.no=i
                                ev2.pmax=self.pmax
                                print(f"[{self.title}] in Q broadcast {str(ev2)}")
                                self.data_in.put(ev2)
                            continue
                        elif no!=self.no:
                            # 自分のじゃ無いから返却
                            # print(f"[{self.title}] Q reject {self.no} {str(ev)}")
                            self.data_in.put(ev)
                            continue
                        else:
                            print(f"[{self.title}] in Q accept {self.no} {str(ev)}")
                    # ---
                    self.output_next_seq = ev.seq
                    return ev
            except Empty:
                return None

        # キューを処理する
        if len(self.output_queue)>0 and self.output_queue[0][0]==self.output_next_seq+1:
            seq, ev = heappop(self.output_queue)
            self.output_next_seq=seq
            if ev is not None:
                if dbg:
                    print(f"[{self.title}] Q POP  {str(ev)}")
                return ev

        while True:

            # get next data
            try:
                ev:Ev = self.data_in.get( timeout=0.5 )
            except Empty:
                return None
            # pass through if seq<=0
            if ev.seq<=0:
                if dbg:
                    print(f"[{self.title}] Q pass through {str(ev)}")
                return ev

            if self.output_next_seq+1==ev.seq:
                # 順番が一致していればそのままコール
                self.output_next_seq=ev.seq
                if dbg:
                    print(f"[{self.title}] Q seq  {str(ev)}")
                return ev
            else:
                # 順番が来てなければキューに入れる
                if dbg:
                    print(f"[{self.title}] Q push {str(ev)}")
                heappush( self.output_queue, (ev.seq,ev) )

    def _event_loop(self):

        xwait = {}
        while not self.req_brake:
            try:
                ev:Ev = self._get_next_data()
                if ev is None:
                    continue

                self.debug(f"Ev {ev}")
                if not isinstance(ev.typ,int):
                    ev.typ = Ev.EndOfData
                try:
                    if ev.typ == Ev.Config:
                        self._event_configure(ev)
                    else:
                        self.proc(ev)
                except:
                    traceback.print_exc()

                try:
                    if ev.typ == Ev.EndOfData or ev.typ == Ev.Stop:
                        if isinstance(ev.no,int) and ev.no>0:
                            pmax = ev.pmax if isinstance(ev.pmax,int) and ev.pmax>0 else 1
                            xwait[f"{ev.no}"]='x'
                            if len(xwait)>=pmax:
                                print(f"+++[{self.title}] Q {len(xwait)} accept {str(ev)}")
                                self.proc_output_event(ev)
                                break
                            else:
                                print(f"+++[{self.title}] Q {len(xwait)} ignore {str(ev)}")
                        else:
                            self.proc_output_event(ev)
                            break
                except:
                    traceback.print_exc()
                    break

            except:
                traceback.print_exc()
                break
        self.stop()
        print(f"[{self.title}] End")

    def _event_start(self):

        print(f"[{self.title}] Load")
        self.load()

        if self.enable_in:
            print(f"[{self.title}] Event Start")
            self._event_loop()
        else:
            print(f"[{self.title}] Process Start")
            self.proc(None)
            self.output(Ev.EndOfData)
            self.stop()

    def output(self, cmd, *args, **kwargs):
        if isinstance(self.data_out,Queue):
            ev = Ev(self.seq_count, cmd, *args, **kwargs )
            self.proc_output_event(ev)

    def proc_output_event(self, ev:Ev):
        if isinstance(ev,Ev):
            ev.seq = self.seq_count
            self.seq_count += 1
            self.output_ev(ev)

    def output_ev(self, ev:Ev):
        if isinstance(self.data_out,Queue):
            if self.no>0:
                ev.no=self.no
                ev.pmax=self.pmax
                if ev.typ==Ev.EndOfData:
                    print(f"&&&[{self.title}] Q output {str(ev)}")
            self.data_out.put(ev)

    def output_ctl(self, ev:Ev):
        if isinstance(self.ctl_out,Queue):
            if self.no>0:
                ev.no=self.no
                ev.pmax=self.pmax
            self.ctl_out.put(ev)

    def configure(self,key,val):
        return

    def to_dict(self):
        return {}

    def load(self ):
        raise NotImplementedError()

    def proc(self, ev ):
        raise NotImplementedError()

    def stop(self):
        return

class VProcess(Process):

    def _dummy():
        pass

    def __init__(self, cls:type[VFunction], data_in:Queue, data_out:Queue, ctl_out:Queue, *args, **kwargs ):
        super().__init__( target=VProcess._dummy, daemon=True )
        self.cls:type[VFunction] = cls
        self.data_in:Queue = data_in
        self.data_out:Queue = data_out
        self.ctl_out:Queue = ctl_out
        self.args = args
        self.kwargs = kwargs
        self.__config_seq:int = 0

    def run(self):
        instance = self.cls( self.data_in, self.data_out, self.ctl_out,  *self.args, **self.kwargs )
        instance._event_start()

    def _configure(self, **kwargs ):
        ev = Ev(  self.__config_seq, Ev.Config, **kwargs )
        self.data_in.put( ev )


class PipeA(VFunction):

    def proc(self, ev ):
        if ev.cmd == Ev.Audio and len(ev.args)>0:
            audio:np.ndarray = ev.args[0]
            # print(f"[{self.__class__.__name__}] {ev.seq} {ev.cmd} {audio.shape} {audio.dtype}")
            audio2 = audio + 0.1
            self.output( Ev.Audio, audio2 )

class PipeB(VFunction):
    def proc(self, ev ):
        if ev.cmd == Ev.Audio and len(ev.args)>0:
            audio:np.ndarray = ev.args[0]
            # print(f"[{self.__class__.__name__}] {ev.seq} {ev.cmd} {audio.shape} {audio.dtype}")
            audio2 = audio + 0.1
            self.output( Ev.Audio, audio2 )

class PipeC(VFunction):
    def proc(self, ev ):
        if ev.cmd == Ev.Audio and len(ev.args)>0:
            audio:np.ndarray = ev.args[0]
            # print(f"[{self.__class__.__name__}] {ev.seq} {ev.cmd} {audio.shape} {audio.dtype}")
            audio2 = audio + 0.1
            self.output( Ev.Audio, audio2 )

def main():

    c1:Queue = Queue()

    q1:Queue = Queue()
    q2:Queue = Queue()
    q3:Queue = Queue()
    q4:Queue = Queue()

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
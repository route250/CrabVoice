from logging import getLogger
import numpy as np
import pandas as pd

logger = getLogger('SttData')
   
def _to_npy_str(value) -> np.ndarray:
    if value is None:
        return None
    return np.frombuffer( bytes(str(value), 'utf-8'), dtype=np.uint8)

def _from_npy_str(npy: np.ndarray) -> str:
    if not isinstance(npy,np.ndarray) or len(npy.shape)!=1:
        return None
    return npy.tobytes().decode('utf-8')

def _to_npy_i16(value) ->np.ndarray:
    if not isinstance(value,(int,float)):
        return None
    return np.array([value]).astype(np.int16)

def _to_npy_i64(value) ->np.ndarray:
    if not isinstance(value,(int,float)):
        return None
    return np.array([value]).astype(np.int64)

def _to_npy_f32(value) ->np.ndarray:
    if not isinstance(value,(int,float)):
        return None
    return np.array([value]).astype(np.float32)

def _from_npy_int( npy:np.ndarray, default=0 ) ->int:
    try:
        return int( npy[0] )
    except:
        return default

def _from_npy_float(npy:np.ndarray,default=0.0) ->float:
    try:
        return float( npy[0] )
    except:
        return default

class SttData:
    """
    音声認識用の音声データ保存クラス
    """
    Start:int = 1
    PreSegment:int= 4
    Segment:int = 5
    PreVoice:int = 6
    Voice:int = 7
    PreText:int=8
    Text:int=9
    Term:int=100
    Dump:int=700

    def __init__(self, typ:int, utc:float, start:int, end:int, sample_rate:int, raw=None, audio=None, hists=None, content:str=None, tag:str=None, seq=0):
        self.utc:float = float(utc)
        self.seq:int = int(seq)
        self.typ:int = int(typ)
        self.start:int = int(start)
        self.end:int = int(end)
        self.sample_rate:int = int(sample_rate)
        self.raw:np.ndarray = raw
        self.audio:np.ndarray = audio
        self.hists:pd.DataFrame = hists
        self.content:str = content
        self.tag:str = tag

    @staticmethod
    def type_to_str(typ:int):
        if SttData.Start==typ:
            return "Start"
        elif SttData.PreSegment==typ:
            return "PreSegment"
        elif SttData.Segment==typ:
            return "Segment"        
        elif SttData.PreVoice==typ:
            return "PreVoice"        
        elif SttData.Voice==typ:
            return "Voice"        
        elif SttData.PreText==typ:
            return "PreText"        
        elif SttData.Text==typ:
            return "Text"        
        elif SttData.Term==typ:
            return "Term"        
        elif SttData.Dump==typ:
            return "Dump"        

    def __str__(self) ->str:
        st_sec = self.start/self.sample_rate
        ed_sec = self.end/self.sample_rate
        return f"[ #{self.seq}, {SttData.type_to_str(self.typ)}, {self.start}({st_sec:.3f}), {self.end}({ed_sec:.3f}) {self.content} ]"

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f'Key must be a string, not {type(key).__name__}')
        if key == 'audio':
            return self.audio.copy() if self.audio is not None else None
        if key == 'raw':
            return self.raw.copy() if self.raw is not None else None
        if self.hists is not None and key in self.hists.columns:
            return self.hists[key].to_numpy()
        raise KeyError(f'Key {key} not found')

    def save(self, out ):
        kwargs = {
            'utc':_to_npy_i64(self.utc),
            'seq':_to_npy_i64(self.seq),
            'typ':_to_npy_i16(self.typ),
            'start':_to_npy_i64(self.start),
            'end':_to_npy_i64(self.end),
            'sample_rate':_to_npy_i64(self.sample_rate),
        }
        if isinstance(self.content,str) and len(self.content)>0:
            kwargs['content'] = _to_npy_str(self.content)
        if isinstance(self.tag,str) and len(self.tag)>0:
            kwargs['tag'] = _to_npy_str(self.tag)
        if self.audio is not None:
            kwargs['audio'] = self.audio
        if self.raw is not None:
            kwargs['raw'] = self.raw
        if self.hists is not None:
            for var in self.hists.columns:
                kwargs[var] = self.hists[var].to_numpy()
        try:
            np.savez_compressed( out, **kwargs )
        except:
            logger.exception(f'can not save to {out}')

    @staticmethod
    def load(path):
        try:
            data = np.load(path, allow_pickle=True)

            seq = _from_npy_int( data.get('seq') )
            typ = _from_npy_int( data.get('typ'), SttData.Dump )
            utc = _from_npy_int( data.get('utc') )
            start = _from_npy_int( data.get('start') )
            end = _from_npy_int( data.get('end') )
            sample_rate = _from_npy_int( data.get('sample_rate') )
            content = _from_npy_str( data.get('content') )
            tag = _from_npy_str( data.get('tag') )
            audio = data['audio'] if 'audio' in data else None
            raw = data['raw'] if 'raw' in data else None
            
            hists_columns = [col for col in data.files if col not in ['seq', 'typ', 'utc', 'start', 'end', 'sample_rate', 'content', 'tag', 'audio', 'raw']]
            hists_data = {col: data[col] for col in hists_columns}
            hists = pd.DataFrame(hists_data) if len(hists_data)>0 else None
            
            return SttData(typ=typ, utc=utc, start=start, end=end, sample_rate=sample_rate, audio=audio, raw=raw, hists=hists, content=content, tag=tag, seq=seq)
        except:
            logger.exception(f'can not load from {path}')
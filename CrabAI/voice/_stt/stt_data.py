
import numpy as np

class SttData:
    Start:int = 1
    PreSegment:int= 4
    Segment:int = 5
    PreVoice:int = 6
    Voice:int = 7
    PreText:int=8
    Text:int=9
    Term:int=100

    def __init__(self, typ:int, start:int, end:int, sample_rate:int, audio=None, hists=None, content:str=None):
        self.seq:int = 0
        self.typ:int = typ
        self.start:int = start
        self.end:int = end
        self.sample_rate:int = sample_rate
        self.audio:np.ndarray = audio
        self.hists:np.ndarray = hists
        self.content:str = content

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

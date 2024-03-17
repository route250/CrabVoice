import json

PRE_KEY=1
IN_KEY=2
AFTER_KEY=3
PRE_VALUE=4
IN_QSTR=5
AFTER_VALUE=6
IN_NUMBER=7
IN_NULL=8
FREE_STR=9
END=999

class JsonStreamParseError(ValueError):
    def __init__(self,msg,pos):
        super().__init__(msg)
        self.pos=pos

class JsonStreamParser:
    """ストリーミングでパースできるJSONパーサ簡易版"""
    def __init__(self):
        self._pos=0
        self._stack = []
        self._phase=PRE_VALUE
        self._obj=None
        self._key=None
        self._val=None
        self._esc=0
        self._ucode=""
        self._lines=1
        self._cols=1

    def _push(self, new_obj, new_phase ):
        if self._obj is None:
            self._obj = new_obj
        elif isinstance(self._obj,dict):
            self._obj[self._key] = new_obj
        else:
            self._obj.append(new_obj)
        self._stack.append( (self._phase,self._obj,self._key) )
        self._phase = new_phase
        self._obj = new_obj
        self._key=None
        self._val=None

    def _pop(self):
        if self._stack:
            self._phase, self._obj, self._key = self._stack.pop()
            self._val = None
            return True
        return False

    def get(self):
        return self._stack[0][1] if self._stack else self._obj

    def put(self, text ):
        if text:
            for cc in text:
                self._put_char(cc)
        return self.get()
    
    def _put_char( self, cc ):
        try:
            if self._esc==0 and cc=="\\" and ( self._phase==IN_QSTR or self._phase==IN_KEY):
                self._esc=1
                return
            elif self._esc==1:
                if "r"==cc:
                    cc="\r"
                elif "n"==cc:
                    cc="\n"
                elif "t"==cc:
                    cc="\t"
                elif "\""==cc:
                    cc="\"x"
                elif "\\"==cc:
                    cc="\\"
                elif "u"==cc:
                    self._esc=2
                    self._ucode="\\u"
                    return
                else:
                    raise JsonStreamParseError(f"invalid escape secence \"\\{cc}",self._pos)
                self._esc=0
            elif self._esc>=2:
                self._ucode+=cc
                if len(self._ucode)<6:
                    return
                self._esc=0
                try:
                    cc = self._ucode.encode().decode('unicode-escape')
                except:
                    raise JsonStreamParseError(f"invalid escape secence \"{self._ucode}",self._pos)
                self._ucode=""

            if self._phase==PRE_KEY:
                # pre key
                if cc=="\"":
                    self._phase=IN_KEY
                    self._key=""
                elif cc=="}":
                    self._put_after_value(cc)
                elif cc>" ":
                    raise JsonStreamParseError(f"Expecting property name enclosed in double quotes: line {self._lines} column {self._cols} (char {self._pos})",self._pos)
            elif self._phase==IN_KEY:
                # in key
                if cc=="\"":
                    self._phase=AFTER_KEY
                else:
                    self._key += cc[0]
            elif self._phase==AFTER_KEY:
                # after key
                if cc==":":
                    self._obj[self._key] = None
                    self._phase=PRE_VALUE
                elif cc>" ":
                    raise JsonStreamParseError(f"invalid char in after key \"{cc}\"",self._pos)
            elif self._phase==PRE_VALUE:
                # pre value
                if cc=="{":
                    self._push( {}, PRE_KEY)
                elif cc=="}" and isinstance(self._obj,dict):
                    self._put_after_value(cc)
                elif cc=="[":
                    self._push( [], PRE_VALUE )
                elif cc=="]" and isinstance(self._obj,list):
                    self._put_after_value(cc)
                elif cc=="\"":
                    self._phase=IN_QSTR
                    self._val=""
                    if isinstance(self._obj,dict):
                        self._obj[self._key] = self._val
                    else:
                        self._obj.append(self._val)
                elif cc=="+" or cc=="-" or cc=="." or "0"<=cc and cc<="9":
                    self._phase=IN_NUMBER
                    self._val=cc
                    if isinstance(self._obj,dict):
                        self._obj[self._key] = None
                    else:
                        pass
                elif cc=="n":
                    self._phase=IN_NULL
                    self._val=cc
                    if isinstance(self._obj,dict):
                        self._obj[self._key] = None
                    else:
                        pass
                elif cc>" ":
                    if self._obj is None and self._key is None and self._val is None:
                        self._phase = FREE_STR
                        self._obj = cc
                    else:
                        raise JsonStreamParseError(f"invalid char in before value \"{cc}\"",self._pos)
            elif self._phase==IN_QSTR:
                # in value
                if cc=="\"":
                    self._phase=AFTER_VALUE
                    self._key=None
                    self._val=None
                else:
                    self._val += cc[0]
                    if isinstance(self._obj,dict):
                        self._obj[self._key] = self._val
                    else:
                        self._obj[-1] = self._val
            elif self._phase==IN_NUMBER:
                # in number
                if cc=="." or cc=="+" or cc=="-" or cc=="e" or "0"<=cc and cc<="9":
                    self._val+=cc
                elif cc<=" " or cc=="," or cc=="}" or cc=="]":
                    num = JsonStreamParser.parse_number(self._val)
                    if isinstance(self._obj,dict):
                        self._obj[self._key] = num
                    else:
                        self._obj.append(num)
                    self._phase=AFTER_VALUE
                    self._key=None
                    self._val=None
                    self._put_after_value(cc)
                else:
                    raise JsonStreamParseError(f"invalid char in number value \"{cc}\"",self._pos)
            elif self._phase==IN_NULL:
                if cc=="u" or cc=="l":
                    self._val+=cc
                elif cc<=" " or cc=="," or cc=="}" or cc=="]":
                    if isinstance(self._obj,dict):
                        pass
                    else:
                        self._obj.append(None)
                    self._phase=AFTER_VALUE
                    self._key=None
                    self._val=None
                    self._put_after_value(cc)
                else:
                    raise JsonStreamParseError(f"invalid char in number value \"{cc}\"",self._pos)

            elif self._phase==AFTER_VALUE:
                # after value
                self._put_after_value(cc)

            elif self._phase==FREE_STR:
                self._obj += cc

            elif self._phase==END:
                # end
                if cc>" ":
                    raise JsonStreamParseError(f"invalid char in after value \"{cc}\"",self._pos)
            else:
                raise JsonStreamParseError(f"invalid phase {self._phase} \"{cc}\"",self._pos)
        finally:
            self._pos+=1
            self._cols+=1
            if cc=="\n":
                self._lines+=1
                self._cols=1

    def _put_after_value(self,cc):
        # after value
        if cc==",":
            if isinstance(self._obj,dict):
                self._phase = PRE_KEY
            elif isinstance(self._obj,list):
                self._phase = PRE_VALUE
            else:
                raise JsonStreamParseError()
        elif cc=="}":
            if self._pop():
                self._phase = AFTER_VALUE
            else:
                self._phase = END
        elif cc=="]":
            if self._pop():
                self._phase = AFTER_VALUE
            else:
                self._phase = END
        elif cc>" ":
            raise JsonStreamParseError(f"invalid char in after value \"{cc}\"",self._pos)

    @staticmethod
    def parse_number( text ):
        if not text:
            return 0
        if "." in text:
            try:
                return float(text)
            except:
                return 0.0
        else:
            try:
                return int(text)
            except:
                return 0


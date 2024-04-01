import sys,os
sys.path.append(os.getcwd())
import unittest

import numpy as np
from io import BytesIO

from CrabAI.voice._stt.stt_data import SttData, _to_npy_str
from CrabAI.voice._stt.stt_data import _to_npy_str, _from_npy_str
from CrabAI.voice._stt.stt_data import _to_npy_i16,_to_npy_i64, _from_npy_int
from CrabAI.voice._stt.stt_data import _to_npy_f32,_to_npy_i16, _from_npy_float

class TestSttDataMethods(unittest.TestCase):

    def test_to_npy_str(self):
        test_case_list=( 'abc123', 'あいうえお', '', None )
        for test_case in test_case_list:
            npy_str = _to_npy_str(test_case)
            back_to_str = _from_npy_str(npy_str)
            self.assertEqual( test_case, back_to_str)

    def test_to_npy_i16(self):
        test_case_list=( 0, 100, -100, None )
        for test_case in test_case_list:
            # print(f"_to_npy_i16 {test_case}")
            tmp_value = _to_npy_i16(test_case)
            back_to_value = _from_npy_int(tmp_value)
            actual = test_case if test_case is not None else 0
            self.assertEqual( actual, back_to_value)

    def test_to_npy_i64(self):
        test_case_list=( 0, 100, -100, None )
        for test_case in test_case_list:
            # print(f"_to_npy_i64 {test_case}")
            tmp_value = _to_npy_i64(test_case)
            back_to_value = _from_npy_int(tmp_value)
            actual = test_case if test_case is not None else 0
            self.assertEqual( actual, back_to_value)

    def test_to_npy_f32(self):
        test_case_list=( 0, 100, -100, None )
        for test_case in test_case_list:
            print(f"_to_npy_f32 {test_case}")
            tmp_value = _to_npy_f32(test_case)
            back_to_value = _from_npy_float(tmp_value)
            actual = test_case if test_case is not None else 0.0
            self.assertEqual( actual, back_to_value)

    def test_to_stt_data(self):
        test_case_list=( 0, 100, -100 )
        for test_case in test_case_list:
            print(f"_to_stt_data {test_case}")
            
            typ=int(test_case) if test_case is not None else None
            utc=int(test_case+1) if test_case is not None else None
            start=int(test_case+2) if test_case is not None else None
            end=int(test_case+3) if test_case is not None else None
            samplerate=int(test_case+4) if test_case is not None else None
            raw = np.array([ test_case+5, test_case+6, test_case+7]) if test_case is not None else None
            audio = np.array([ test_case+8, test_case+9, test_case+10]) if test_case is not None else None
            hists = None
            content = f"{test_case+20}" if test_case is not None else None
            tag = f"{test_case+21}" if test_case is not None else None
            
            tmp_value = SttData( typ=typ, utc=utc, start=start, end=end, sample_rate=samplerate, raw=raw, audio=audio, hists=hists, content=content, tag=tag )

            self.assertEqual( typ, tmp_value.typ )
            self.assertEqual( utc, tmp_value.utc )
            self.assertEqual( start, tmp_value.start )
            self.assertEqual( end, tmp_value.end )
            self.assertEqual( samplerate, tmp_value.sample_rate )

            xxxx = BytesIO()
            tmp_value.save( xxxx )
            xxxx.seek(0)
            
            back_to_value = SttData.load(xxxx)

            self.assertEqual( typ, back_to_value.typ )
            self.assertEqual( utc, back_to_value.utc )
            self.assertEqual( start, back_to_value.start )
            self.assertEqual( end, back_to_value.end )
            self.assertEqual( samplerate, back_to_value.sample_rate )

            # actual = test_case if test_case is not None else 0.0
            # self.assertEqual( actual, back_to_value)

if __name__ == '__main__':
    unittest.main()
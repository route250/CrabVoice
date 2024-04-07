import sys,os
sys.path.append(os.getcwd())
import unittest

import unittest
from CrabAI.voice._stt.ring_buffer import RingBuffer
import numpy as np

class TestRingBuffer(unittest.TestCase):

    def test_initialization(self):
        rb = RingBuffer(10, dtype=np.float64)
        self.assertEqual(rb.capacity, 10)
        self.assertEqual(rb.dtype, np.float64)
        self.assertEqual(len(rb), 0)

    def test_add_append(self):
        rb = RingBuffer(5)
        # 最初の一個を追加
        rb.add(1.0)
        self.assertEqual(len(rb), 1)
        self.assertTrue(np.array_equal(rb.to_numpy(), np.array([1.0], dtype=np.float32)))

        # 溢れない範囲で追加
        rb.append(np.array([2.0, 3.0]))
        self.assertEqual(len(rb), 3)
        self.assertTrue(np.array_equal(rb.to_numpy(), np.array([1.0,2.0,3.0], dtype=np.float32)))

        # ちょっと溢れる追加
        rb.append(np.array([4.0, 5.0, 6.0]))
        self.assertEqual(len(rb), 5)
        self.assertTrue(np.array_equal(rb.buffto_numpy(), np.array([2.0,3.0,4.0, 5.0, 6.0], dtype=np.float32)))

        # 完全に溢れる追加
        rb.append(np.array([10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0]))
        self.assertEqual(len(rb), 5)
        self.assertTrue(np.array_equal(rb.buffto_numpy(), np.array([13.0,14.0, 15.0, 16.0, 17.0], dtype=np.float32)))

    def test_remove(self):
        rb = RingBuffer(5)
        rb.append(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        rb.remove(2)
        self.assertEqual(len(rb), 3)
        self.assertTrue(np.array_equal(rb.to_numpy(), np.array([3.0, 4.0, 5.0], dtype=np.float32)))

    def test_clear(self):
        rb = RingBuffer(5)
        rb.append(np.array([1.0, 2.0, 3.0]))
        rb.clear()
        self.assertEqual(len(rb), 0)

    def test_is_full(self):
        rb = RingBuffer(3)
        self.assertFalse(rb.is_full())
        rb.append(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(rb.is_full())

    def test_get_set(self):
        rb = RingBuffer(3)
        rb.append(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(rb.get(1), 2.0)
        rb.set(1, 4.0)
        self.assertEqual(rb.get(1), 4.0)

    def test_to_numpy(self):
        rb = RingBuffer(5)
        rb.append(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_array_equal(rb.to_numpy(1, 4), np.array([2.0, 3.0, 4.0], dtype=np.float32))

    def test_getitem(self):
        rb = RingBuffer(5)
        rb.append(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(rb[2], 3.0)
        np.testing.assert_array_equal(rb[1:4], np.array([2.0, 3.0, 4.0], dtype=np.float32))

    def test_ma(self):
        np_vad = np.random.random_sample( 100 )
        np_vad_slope = np.zeros_like( np_vad )
        window = 5
        rb_vad = RingBuffer(20)
        rb_vad_slope = RingBuffer(20)

        for i in range(len(np_vad)):
            offset = window//2 - window
            j = i + offset
            new_value = np_vad[i]
            np_start = max( 0, i-window)
            np_frame = np_vad[np_start:i+1]
            np_ma = np_frame.mean()
            if j>=0:
                np_vad_slope[j] = np_ma

            rb_vad.add( new_value )
            rb_vad_slope.add( new_value )
            rb_frame = rb_vad.to_numpy(-window)
            rb_ma = rb_frame.mean()
            print( f"{np_ma} {rb_ma}")
            if len(rb_vad_slope)+offset>=0:
                rb_vad_slope.set(len(rb_vad_slope)+offset, rb_ma)


if __name__ == '__main__':
    unittest.main()

"""Policy boundary and actual WFDB round-trip tests using synthetic signals."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import wfdb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ecg_lbbb.process_metadata import classify
from ecg_lbbb.extract_v6 import load_v6


class ProcessingTests(unittest.TestCase):
    def test_policy(self):
        self.assertEqual(classify({'CLBBB': 100, 'SR': 0})[0], 1)
        self.assertEqual(classify({'NORM': 100})[0], 0)
        for codes in [{'CLBBB': 80}, {'NORM': 100, 'CLBBB': 100},
                      {'NORM': 100, 'ILBBB': 0}, {'CLBBB': 100, 'LVH': 0},
                      {'NORM': 100, 'PVC': 0}, {'SR': 0}]:
            self.assertIsNone(classify(codes)[0])

    def test_wfdb_v6_by_name_and_rate_guard(self):
        with tempfile.TemporaryDirectory() as root:
            signal = np.column_stack([np.linspace(-1, 1, 1000), np.zeros(1000)])
            for name, fs in [('valid', 100), ('wrong_rate', 500)]:
                wfdb.wrsamp(name, fs=fs, units=['mV', 'mV'], sig_name=['V6', 'I'],
                            p_signal=signal, fmt=['16', '16'], write_dir=root)
            x = load_v6(Path(root) / 'valid')
            self.assertEqual(x.shape, (1, 1000))
            self.assertEqual(x.dtype, np.float32)
            np.testing.assert_allclose(x[0], signal[:, 0], atol=0.0001)
            with self.assertRaises(ValueError):
                load_v6(Path(root) / 'wrong_rate')


if __name__ == '__main__':
    unittest.main()

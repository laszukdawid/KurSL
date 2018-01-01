import unittest
import numpy as np

from kursl import Preprocessor

class TestPreprocessor(unittest.TestCase):

    @staticmethod
    def peak_norm(t, t0, amp, s):
        _t = (t-t0)/s
        return amp*np.exp(-_t*_t)

    @staticmethod
    def peak_triangle(t, t0, amp, s):
        x = amp*(1-np.abs((t-t0)/s))
        x[x<0] = 0
        return x

    @staticmethod
    def peak_lorentz(t, t0, amp, s):
        _t = (t-t0)/(0.5*s)
        return amp/(_t*_t + 1)

    def test_remove_peak_norm(self):
        """Test removing single peak as a Gaussian peak"""
        #If too small segment then it doesn't converge properly
        t = np.arange(0, 2, 0.001)

        param_in = np.array([1.2, 0.4, 0.1])
        peak_in = self.peak_norm(t, param_in[0], param_in[1], param_in[2])

        peak_out, param_out = Preprocessor._remove_peak(t, peak_in, ptype='norm')
        self.assertTrue(np.allclose(param_in, param_out),
                            "Parameters should be the same")
        self.assertTrue(np.allclose(peak_in, peak_out),
                            "Peaks should be the same")

    def test_remove_peak_triangle(self):
        """Test removing single peak as a triangle peak"""
        t = np.arange(0, 2, 0.001)

        param_in = np.array([0.5, 1.5, 0.3])
        peak_in = self.peak_triangle(t, param_in[0], param_in[1], param_in[2])

        peak_out, param_out = Preprocessor._remove_peak(t, peak_in, ptype="triang")
        self.assertTrue(np.allclose(param_in, param_out),
                            "Parameters should be the same")
        self.assertTrue(np.allclose(peak_in, peak_out),
                            "Peaks should be the same")

    def test_remove_peak_lorentz(self):
        """Test removing single peak as a Lorentzian peak"""
        t = np.arange(0, 3, 0.001)
        param_in = np.array([0.8, 2.5, 0.2])
        peak_in = self.peak_lorentz(t, param_in[0], param_in[1], param_in[2])

        peak_out, param_out = Preprocessor._remove_peak(t, peak_in, ptype="lorentz")
        self.assertTrue(np.allclose(param_in, param_out),
                            "Parameters should be the same")
        self.assertTrue(np.allclose(peak_in, peak_out),
                            "Peaks should be the same")

    def test_remove_peak_unknown_value(self):
        """Test removing single peak with unknown peak name.
        It should throw ValueError exception."""

        t = np.arange(0, 2, 0.001)
        peak_in = np.random.random(t.size)
        with self.assertRaises(ValueError) as context:
            Preprocessor._remove_peak(t, peak_in, ptype="unknown")
            Preprocessor._remove_peak(t, peak_in, ptype="other")

    def test_remove_energy_default(self):
        t = np.arange(0, 5, 0.001)
        p1 = [1.0, 2.0, 0.4]
        p2 = [3.5, 1.5, 0.3]
        p3 = [2.0, 1.0, 0.2]

        peak1 = self.peak_norm(t, p1[0], p1[1], p1[2]) # Biggest
        peak2 = self.peak_lorentz(t, p2[0], p2[1], p2[2]) # Middle
        peak3 = self.peak_triangle(t, p3[0], p3[1], p3[2]) # Smallest
        s = peak1 + peak2 + peak3

        preprocessor = Preprocessor()
        self.assertEqual(preprocessor.energy_ratio, 0.1, "Default energy ratio is 0.1")
        s_res, p_out = preprocessor.remove_energy(t, s)

        # Check number of peaks
        self.assertEqual(len(p_out), 3, "Expected 3 peaks by default")

        # Checking location
        msg = "Expected: {:.3}, Receive: {:.3}"
        self.assertTrue(abs(1.00-p_out[0,0])<0.01, msg.format(1.00, p_out[0,0]))
        self.assertTrue(abs(3.50-p_out[1,0])<0.01, msg.format(3.50, p_out[1,0]))
        self.assertTrue(abs(2.00-p_out[2,0])<0.01, msg.format(2.00, p_out[2,0]))

        # Checking amplitude
        self.assertTrue(abs(1.99-p_out[0,1])<0.01, msg.format(1.99, p_out[0,1]))
        self.assertTrue(abs(1.33-p_out[1,1])<0.01, msg.format(1.33, p_out[1,1]))
        self.assertTrue(abs(0.93-p_out[2,1])<0.01, msg.format(0.93, p_out[2,1]))

        # Checking width
        self.assertTrue(abs(0.40-p_out[0,2])<0.01, msg.format(0.40, p_out[0,2]))
        self.assertTrue(abs(0.23-p_out[1,2])<0.01, msg.format(0.23, p_out[1,2]))
        self.assertTrue(abs(0.12-p_out[2,2])<0.01, msg.format(0.12, p_out[2,2]))

    def test_remove_energy_triangle(self):
        t = np.arange(0, 5, 0.001)
        p1 = [1.0, 2.0, 0.4]
        p2 = [3.5, 1.5, 0.3]
        p3 = [2.0, 1.0, 0.2]

        peak1 = self.peak_norm(t, p1[0], p1[1], p1[2]) # Biggest
        peak2 = self.peak_lorentz(t, p2[0], p2[1], p2[2]) # Middle
        peak3 = self.peak_triangle(t, p3[0], p3[1], p3[2]) # Smallest
        s = peak1 + peak2 + peak3

        preprocessor = Preprocessor()
        self.assertEqual(preprocessor.energy_ratio, 0.1, "Default energy ratio is 0.1")
        s_res, p_out = preprocessor.remove_energy(t, s, ptype="triang")

        # Check number of peaks
        self.assertEqual(len(p_out), 3, "Expected 3 peaks by default")

        # Checking location
        msg = "Expected: {:.3}, Receive: {:.3}"
        self.assertTrue(abs(p1[0]-p_out[0,0])<0.01, msg.format(p1[0], p_out[0,0]))
        self.assertTrue(abs(p2[0]-p_out[1,0])<0.01, msg.format(p2[0], p_out[1,0]))
        self.assertTrue(abs(p3[0]-p_out[2,0])<0.01, msg.format(p3[0], p_out[2,0]))

        # Checking amplitude
        self.assertTrue(abs(2.16-p_out[0,1])<0.02, msg.format(p1[1], p_out[0,1]))
        self.assertTrue(abs(1.43-p_out[1,1])<0.02, msg.format(p2[1], p_out[1,1]))
        self.assertTrue(abs(1.01-p_out[2,1])<0.02, msg.format(p3[1], p_out[2,1]))

        # Checking width
        self.assertTrue(abs(0.65-p_out[0,2])<0.02, msg.format(0.65, p_out[0,2]))
        self.assertTrue(abs(0.37-p_out[1,2])<0.02, msg.format(0.37, p_out[1,2]))
        self.assertTrue(abs(0.20-p_out[2,2])<0.02, msg.format(0.20, p_out[2,2]))

    def test_remove_energy_max_osc(self):
        """Test remove_energy with limited number of peaks"""
        t = np.arange(0, 5, 0.001)
        p1 = [1.0, 2.0, 0.4]
        p2 = [3.5, 1.5, 0.3]
        p3 = [2.0, 1.0, 0.2]

        peak1 = self.peak_norm(t, p1[0], p1[1], p1[2]) # Biggest
        peak2 = self.peak_lorentz(t, p2[0], p2[1], p2[2]) # Middle
        peak3 = self.peak_triangle(t, p3[0], p3[1], p3[2]) # Smallest
        s = peak1 + peak2 + peak3

        max_peaks = 2
        preprocessor = Preprocessor()
        s_res, p_out = preprocessor.remove_energy(t, s, max_peaks=max_peaks)

        # Check number of peaks
        self.assertEqual(len(p_out), 2, "Expected 2 peaks because of flag")

    def test_remove_energy_different_energy_ratio(self):
        """Test remove_energy after changing energy_ratio to different value"""
        t = np.arange(0, 5, 0.001)
        p1 = [1.0, 2.0, 0.4]
        p2 = [3.5, 1.5, 0.3]
        p3 = [2.0, 1.0, 0.2]

        peak1 = self.peak_norm(t, p1[0], p1[1], p1[2]) # Biggest
        peak2 = self.peak_lorentz(t, p2[0], p2[1], p2[2]) # Middle
        peak3 = self.peak_triangle(t, p3[0], p3[1], p3[2]) # Smallest
        s = peak1 + peak2 + peak3

        energy_ratio = 0.7
        preprocessor = Preprocessor()
        s_res, p_out = preprocessor.remove_energy(t, s, ratio=energy_ratio)

        # Check number of peaks
        self.assertEqual(len(p_out), 1, "One peak expected with large energy ratio")


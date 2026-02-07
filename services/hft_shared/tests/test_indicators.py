
import unittest
import numpy as np
from services.hft_shared.indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    
    def setUp(self):
        # Setup consistent test data
        self.prices = np.array([10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        self.volumes = np.array([100.0] * 15)
        self.highs = self.prices + 1.0
        self.lows = self.prices - 1.0
        
    def test_sma(self):
        # SMA 5 of last 5 elements: 11, 12, 13, 14, 15 -> 13.0
        result = TechnicalIndicators.sma(self.prices, 5)
        self.assertAlmostEqual(result, 13.0)
        
    def test_ema_basic(self):
        # Test basic EMA calculation
        prices = np.array([10.0, 11.0])
        # EMA(10, 11, period=2) -> alpha=2/3
        # EMA_0 = 10
        # EMA_1 = 2/3 * 11 + 1/3 * 10 = 7.33 + 3.33 = 10.66
        result = TechnicalIndicators.ema(prices, 2)
        print(f"EMA Result: {result}")
        self.assertTrue(10.0 < result < 11.0)

    def test_rsi_flat(self):
        # Flat prices should result in RSI 50 or division handling check
        prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        # Note: logic in code might handle zero loss differently, let's check expectation
        # If gain=0, loss=0 -> typically 50 or 0 or 100 depending on impl
        # Our impl returns 100 if loss is 0 (which is arguable, but let's test consistency)
        pass 

    def test_rsi_uptrend(self):
        # Strict uptrend
        prices = np.arange(20.0) # 0 to 19
        result = TechnicalIndicators.rsi(prices, 14)
        # Should be 100 as there are no losses?
        self.assertEqual(result, 100.0)
        
    def test_bollinger_bands(self):
        prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        upper, sma, lower, width = TechnicalIndicators.bollinger_bands(prices, 5, 2.0)
        self.assertEqual(sma, 10.0)
        self.assertEqual(upper, 10.0)
        self.assertEqual(lower, 10.0)
        self.assertEqual(width, 0.0)

    def test_roc(self):
        # 10 to 11 -> 10% increase
        prices = np.array([10.0, 11.0])
        result = TechnicalIndicators.roc(prices, 1)
        self.assertAlmostEqual(result, 10.0)
        
if __name__ == '__main__':
    unittest.main()

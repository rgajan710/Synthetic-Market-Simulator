import backtrader as bt

class RSI_Strategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14)

    def next(self):
        if not self.position:
            if self.rsi < 30:
                self.buy()
        elif self.rsi > 70:
            self.sell()
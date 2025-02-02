import pandas as pd
import yfinance as yf
import datetime
import backtrader as bt
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6) # (w, h)


def sim_leverage(proxy, leverage=1, expense_ratio=0.0, initial_value=1.0):
    """
    Simula un ETF con leva in base al proxy, alla leva finanziaria e al rapporto di spesa.

    La variazione percentuale giornaliera viene calcolata prendendo la variazione percentuale
    giornaliera del proxy, sottraendo il rapporto di spesa giornaliera, quindi moltiplicando
    per la leva finanziaria.
    """
    pct_change = proxy.pct_change(1)
    pct_change = (pct_change - expense_ratio / 252) * leverage
    sim = (1 + pct_change).cumprod() * initial_value
    sim.bfill(inplace=True)
    sim[0] = initial_value
    return sim


start = "1986-05-19"
end = "2019-01-01"

vfinx = yf.download("VFINX", start, end)["Adj Close"]
vustx = yf.download("VUSTX", start, end)["Adj Close"]

vfinx.columns=['close']
vustx.columns=['close']
upro_sim = sim_leverage(vfinx, leverage=3, expense_ratio=0.0092)["close"].to_frame()
tmf_sim = sim_leverage(vustx, leverage=3, expense_ratio=0.0109)["close"].to_frame()

for column in ["open", "high", "low"]:
    upro_sim[column] = upro_sim["close"]
    tmf_sim[column] = tmf_sim["close"]

upro_sim["volume"] = 0
tmf_sim["volume"] = 0

upro_sim = bt.feeds.PandasData(dataname=upro_sim)
tmf_sim = bt.feeds.PandasData(dataname=tmf_sim)
vfinx = bt.feeds.YahooFinanceData(dataname="VFINX",
                                  fromdate=datetime.date(1986,5,19),
                                  todate=datetime.date(2019,1,1))

class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.getposition(self.data).size:
            self.order_target_percent(self.data, target=1.0)

def backtest(datas, strategy, plot=False, **kwargs):
    cerebro = bt.Cerebro()
    for data in datas:
        cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addstrategy(strategy, **kwargs)
    results = cerebro.run()
    if plot:
        cerebro.plot()
    return (results[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            results[0].analyzers.returns.get_analysis()['rnorm100'],
            results[0].analyzers.sharperatio.get_analysis()['sharperatio'])


dd, cagr, sharpe = backtest([vfinx], BuyAndHold, plot=True)
print(f"Max Drawdown: {dd:.2f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")

dd, cagr, sharpe = backtest([upro_sim], BuyAndHold)
print(f"Max Drawdown: {dd:.2f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")

dd, cagr, sharpe = backtest([tmf_sim], BuyAndHold)
print(f"Max Drawdown: {dd:.2f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")


class AssetAllocation(bt.Strategy):
    params = (
        ('equity', 0.6),
    )

    def __init__(self):
        self.UPRO = self.datas[0]
        self.TMF = self.datas[1]
        self.counter = 0

    def next(self):
        if self.counter % 20 == 0:
            self.order_target_percent(self.UPRO, target=self.params.equity)
            self.order_target_percent(self.TMF, target=(1 - self.params.equity))
        self.counter += 1

dd, cagr, sharpe = backtest([upro_sim, tmf_sim], AssetAllocation, plot=True, equity=0.6)
print(f"Max Drawdown: {dd:.2f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")

sharpes = {}
for perc_equity in range(0, 101, 5):
    sharpes[perc_equity] = backtest([upro_sim, tmf_sim], AssetAllocation, equity=(perc_equity / 100.0))[2]

series = pd.Series(sharpes)
ax = series.plot(title="UPRO/TMF allocation vs Sharpe")
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("Percent Portfolio UPRO");
print(f"Max Sharpe of {series.max():.3f} at {series.idxmax()}% UPRO")
print("")

dd, cagr, sharpe = backtest([upro_sim, tmf_sim], AssetAllocation, plot=True, equity=0.4)
print(f"Max Drawdown: {dd:.2f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")
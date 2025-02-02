from datetime import datetime
import pandas as pd
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6) # (w, h)
plt.ioff()

tickers = pd.read_csv('spy/tickers.csv', header=None)[1].tolist()
start = datetime(2015, 1, 1)
end = datetime(2020, 1, 1)

datas = [bt.feeds.GenericCSVData(
            fromdate=start,
            todate=end,
            dataname=f"spy/{ticker}.csv",
            dtformat=('%Y-%m-%d'),
            openinterest=-1,
            nullvalue=0.0,
            plot=False
        ) for ticker in tickers]

def backtest(datas, strategy, plot=None, **kwargs):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_coc(True)
    cerebro.broker.setcash(1_000_000)
    for data in datas:
        cerebro.adddata(data)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addstrategy(strategy, **kwargs)
    results = cerebro.run()
    if plot:
        cerebro.plot(iplot=False)[0][0]
    return (results[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            results[0].analyzers.returns.get_analysis()['rnorm100'],
            results[0].analyzers.sharperatio.get_analysis()['sharperatio'])


def max_n(array, n):
    return np.argpartition(array, -n)[-n:]


class CrossSectionalMR(bt.Strategy):
    params = (
        ('num_positions', 100),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]["pct"] = bt.indicators.PercentChange(d.close, period=1)

    def prenext(self):
        self.next()

    def next(self):
        available = list(filter(lambda d: len(d), self.datas))  # usa solo i dati disponibili nella giornata precedente
        rets = np.zeros(len(available))
        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]

        market_ret = np.mean(rets)
        weights = -(rets - market_ret)
        max_weights_index = max_n(np.abs(weights), self.params.num_positions)
        max_weights = weights[max_weights_index]
        weights = weights / np.sum(np.abs(max_weights))

        for i, d in enumerate(available):
            if i in max_weights_index:
                self.order_target_percent(d, target=weights[i])
            else:
                self.order_target_percent(d, 0)

dd, cagr, sharpe = backtest(datas, CrossSectionalMR, plot=True, num_positions=100)
print(f"Max Drawdown: {dd:.2f}%\nAPR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")


def min_n(array, n):
    return np.argpartition(array, n)[:n]


def max_n(array, n):
    return np.argpartition(array, -n)[-n:]


class CrossSectionalMR(bt.Strategy):
    params = (
        ('n', 100),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]["pct"] = bt.indicators.PercentChange(d.close, period=5)
            self.inds[d]["std"] = bt.indicators.StandardDeviation(d.close, period=5)

    def prenext(self):
        self.next()

    def next(self):
        available = list(filter(lambda d: len(d) > 5, self.datas)) # usa solo i dati disponibili nella giornata precedente
        rets = np.zeros(len(available))
        stds = np.zeros(len(available))
        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]
            stds[i] = self.inds[d]['std'][0]

        market_ret = np.mean(rets)
        weights = -(rets - market_ret)
        max_weights_index = max_n(np.abs(weights), self.params.n)
        low_volality_index = min_n(stds, self.params.n)
        selected_weights_index = np.intersect1d(max_weights_index,
                                                low_volality_index)
        if not len(selected_weights_index):
            # non si effettua trade
            return

        selected_weights = weights[selected_weights_index]
        weights = weights / np.sum(np.abs(selected_weights))
        for i, d in enumerate(available):
            if i in selected_weights_index:
                self.order_target_percent(d, target=weights[i])
            else:
                self.order_target_percent(d, 0)

dd, cagr, sharpe = backtest(datas, CrossSectionalMR, plot=True, n=100)
print(f"Max Drawdown: {dd:.2f}%\nAPR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")
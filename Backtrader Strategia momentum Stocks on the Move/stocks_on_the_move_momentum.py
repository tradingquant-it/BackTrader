from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (10, 6)  # (w, h)
plt.ioff()
tickers = pd.read_csv('data/tickers.csv', header=None)[1].tolist()
stocks = (
    (pd.concat(
        [pd.read_csv(f"data/{ticker}.csv", index_col='date', parse_dates=True)[
             'close'
         ].rename(ticker)
         for ticker in tickers],
        axis=1,
        sort=True)
    )
)
stocks = stocks.loc[:, ~stocks.columns.duplicated()]

from scipy.stats import linregress


def momentum(closes):
    returns = np.log(closes)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    return ((1 + slope) ** 252) * (rvalue ** 2)  # Annualizza la pendenza e moltiplica per R^2


momentums = stocks.copy(deep=True)
for ticker in tickers:
    momentums[ticker] = stocks[ticker].rolling(90).apply(momentum, raw=False)

plt.figure(figsize=(12, 9))
plt.xlabel('Days')
plt.ylabel('Stock Price')

bests = momentums.max().sort_values(ascending=False).index[:5]
for best in bests:
    end = momentums[best].index.get_loc(momentums[best].idxmax())
    rets = np.log(stocks[best].iloc[end - 90: end])
    x = np.arange(len(rets))
    slope, intercept, r_value, p_value, std_err = linregress(x, rets)
    plt.plot(np.arange(180), stocks[best][end - 90:end + 90])
    plt.plot(x, np.e ** (intercept + slope * x))

plt.show()


import backtrader as bt


class Momentum(bt.Indicator):
    lines = ('trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        returns = np.log(self.data.get(size=self.p.period))
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + slope) ** 252
        self.lines.trend[0] = annualized * (rvalue ** 2)


class Strategy(bt.Strategy):
    def __init__(self):
        self.i = 0
        self.inds = {}
        self.spy = self.datas[0]
        self.stocks = self.datas[1:]

        self.spy_sma200 = bt.indicators.SimpleMovingAverage(self.spy.close, period=200)
        for d in self.stocks:
            self.inds[d] = {}
            self.inds[d]["momentum"] = Momentum(d.close, period=90)
            self.inds[d]["sma100"] = bt.indicators.SimpleMovingAverage(d.close, period=100)
            self.inds[d]["atr20"] = bt.indicators.ATR(d, period=20)

    def prenext(self):
        # chiamata al metodo next() anche quando non ci sono i dati disponibili per tutti i ticker
        self.next()

    def next(self):
        if self.i % 5 == 0:
            self.rebalance_portfolio()
        if self.i % 10 == 0:
            self.rebalance_positions()
        self.i += 1

    def rebalance_portfolio(self):
        # filtra i dati per i quali possiamo calcolare l'indicatore
        self.rankings = list(filter(lambda d: len(d) > 100, self.stocks))
        self.rankings.sort(key=lambda d: self.inds[d]["momentum"][0])
        num_stocks = len(self.rankings)

        # vende le azioni che rispettano le condizioni
        for i, d in enumerate(self.rankings):
            if self.getposition(self.data).size:
                if i > num_stocks * 0.2 or d < self.inds[d]["sma100"]:
                    self.close(d)

        if self.spy < self.spy_sma200:
            return

        # acquista azioni con il capitale rimanente
        for i, d in enumerate(self.rankings[:int(num_stocks * 0.2)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            if not self.getposition(self.data).size:
                size = value * 0.001 / self.inds[d]["atr20"]
                self.buy(d, size=size)

    def rebalance_positions(self):
        num_stocks = len(self.rankings)

        if self.spy < self.spy_sma200:
            return

        # ribilanciamento delle azioni
        for i, d in enumerate(self.rankings[:int(num_stocks * 0.2)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            size = value * 0.001 / self.inds[d]["atr20"]
            self.order_target_size(d, size)

cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.set_coc(True)

spy = bt.feeds.YahooFinanceData(dataname='SPY',
                                 fromdate=datetime(2012,2,28),
                                 todate=datetime(2018,2,28),
                                 plot=False)
cerebro.adddata(spy)  # aggiunge l'indice S&P 500

for ticker in tickers:
    df = pd.read_csv(f"survivorship-free/{ticker}.csv",
                     parse_dates=True,
                     index_col=0)
    if len(df) > 100: # i dati devono essere abbastanza per calcolare la SMA a 100 giorni
        cerebro.adddata(bt.feeds.PandasData(dataname=df, plot=False))

cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addstrategy(Strategy)
results = cerebro.run()
cerebro.plot(iplot=False)[0][0]
plt.show()
print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

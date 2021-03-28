#-*- coding: utf-8 -*-
from StockCode.stock import *
from StockCode.self_money import *
import pandas as pd
# pd.core.common.is_list_like = pd.api.types.is_list_like
import numpy as np
from datetime import  datetime
from scipy import stats
import os
import glob

class StockLake():
    '''
    default benchmark ^TWII
    '''
    def __init__(self, start_date = None, end_date = None, benchmark = '^TWII'):
        self.stockList = []
        self.benchmark = [benchmark]
        self.start = start_date
        self.end = end_date
        self.directory_root =  'StockLake'
        self.directory_folder = ['Stock','Benchmark']
        self.df = pd.DataFrame()
        self.df_benchmark = pd.DataFrame()

    def add(self, stock):
        if isinstance(stock, list) == False:
            raise TypeError('input stock code list like [\'stock\']')
        for i in stock:
            if i in self.stockList:
                raise ValueError('Portfolio already have {}'.format(i))
        self.stockList.extend(stock)
        print('Portfolio add ', *self.stockList)

    def load_stock(self):
        for stock in self.stockList:
            df_ = dataFrame(self.start, self.end, str(stock)).df
            df_['Stock'] = stock
            self.df = self.df.append(df_)
            print(str(stock)+" download complete!")
        for stock in self.benchmark:
            df_ = dataFrame(self.start, self.end, str(stock)).df
            df_['Benchmark'] = stock
            self.df_benchmark = self.df_benchmark.append(df_)
            print(str(stock)+" benchmark download complete!")

    def download(self, file = None, benchmark = None):
        if file == None:
            stock = 'StockLake_{}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            benchmark = 'StockLake_{}_{}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S"),'benchmark')
            for folder in self.directory_folder:
                folder_path = os.path.join(self.directory_root,folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            stock_file = os.path.join(self.directory_root,self.directory_folder[0] ,stock)
            benchmark_file = os.path.join(self.directory_root,self.directory_folder[1] , benchmark)
        else:
            stock_file = file
            benchmark_file = benchmark
        self.df.to_csv(stock_file)
        self.df_benchmark.to_csv(benchmark_file)
        print('Download Stock     to : {}'.format(stock_file) )
        print('Download Benchmark to : {}'.format(benchmark_file) )

    def load(self, file = None, benchmark = None):
        if file == None:
            file_ = glob.glob(os.path.join(self.directory_root,self.directory_folder[0],'*.csv'))
            benchmark_ = glob.glob(os.path.join(self.directory_root,self.directory_folder[1],'*.csv'))
            stock = sorted(file_, reverse = True)[0]
            benchmark = sorted(benchmark_, reverse = True)[0]
            if not os.path.exists(stock):
                raise ValueError("{} dose not exist".format(stock))
            if not os.path.exists(benchmark):
                print("{} dose not exist".format(benchmark))
        else:
            stock = file
            benchmark = benchmark
        self.df = pd.read_csv(stock, engine='python',index_col = 0,dtype={'Stock': object})
        self.df_benchmark = pd.read_csv(benchmark, engine='python',index_col = 0,dtype={'Benchmark': object})
        print('Load Stock     data from : {}'.format(stock))
        print('Load Benchmark data from : {}'.format(benchmark))


class Portfolio():
    def __init__(self, start, end, init_money=1000000, riskfree=0):
        self.stockList = []
        self.allocation = []
        self.init_money = init_money
        self.riskfree = riskfree
        self.df = pd.DataFrame()
        self.df_position = pd.DataFrame()
        self.start = start
        self.end = end
        start_date = datetime.strptime(self.start, '%Y/%m/%d')
        end_date = datetime.strptime(self.end, '%Y/%m/%d')
        self.n_days = (end_date - start_date).days

    def add(self, stock):
        if isinstance(stock, list) == False:
            raise TypeError('input stock code list like [\'stock\']')
        for i in stock:
            if i in self.stockList:
                raise ValueError('Portfolio already have {}'.format(i))
        self.stockList.extend(stock)
        self.allocation = [1 / len(self.stockList)] * len(self.stockList)  # default setting equal allocation
        self.pairs = list(zip(self.stockList,self.allocation))
        print('Portfolio add ', *self.stockList)

    def remove(self, stock):
        tmp = []
        if isinstance(stock, list) == False:
            raise TypeError('input stock code list like [\'stock\']')
        for i in stock:
            if i not in self.stockList:
                print('Portfolio do not have {}'.format(i))
            else:
                tmp.append(i)
                self.stockList.remove(i)
        print('Portfolio remove ', *tmp)

    def assignment(self,allocation ):
        if allocation != None:
            self.allocation = allocation
        if len(self.allocation) != len(self.stockList):
            raise ValueError('len stock allocation dose not match len stock list')
        if sum(self.allocation) != 1:
            total = sum(self.allocation)
            temp = self.allocation
            self.allocation = [nominalization/total for nominalization in self.allocation]
            print('allocation sum != 1 so ', *temp, ' nominalization to ',*self.allocation)
        self.pairs = list(zip(self.stockList,self.allocation))

    def now_stockList(self):
        print('Portfolio Stock List : ', self.pairs )

    def refresh(self):
        self.stockList = []
        print('Clear Profolio Sotck List')

    def load_data(self, StockLake=None):
        if self.start < StockLake.start or self.end > StockLake.end:
            raise ValueError(
                'StockLake date below or beyond Portfolio date, please change Stock Lake')
        Stock = StockLake.df
        ## build day filter mask, modify / to -, index.fliter dosen't support /
        mask = (Stock.index >= self.start.replace('/', '-') ) & (Stock.index <= self.end.replace('/', '-'))
        ## filter mask data
        df__ = Stock[mask]
        print('Start load stock data')
        for pair  in self.pairs:
            stock = pair[0]
            allocation = pair[1]
            df_ = df__[df__['Stock'] == stock]
            df_['Stock'] = stock
            df_['ROI'] = df_['Adj Close'] / df_['Adj Close'].iloc[0]
            df_['Allocation'] = df_['ROI'] * allocation
            df_['Position'] = df_['Allocation'] * self.init_money
            self.df = self.df.append(df_)
            self.df_position = pd.concat([self.df_position, df_['Position']], axis=1)
        self.df_position.columns = self.stockList
        self.df_position['Total Position'] = self.df_position.sum(axis=1)
        self.df_position['Daily Return'] = self.df_position['Total Position'].pct_change(1)
        self.cum_return = round(100 * (self.df_position['Total Position'][-1] / self.df_position['Total Position'][0] - 1), 4)
        print('Start load benchmark data')
        benchmark = StockLake.df_benchmark
        mask = (benchmark.index >= self.start.replace('/', '-') ) & (benchmark.index <= self.end.replace('/', '-'))
        self.df_benchmark = benchmark[mask]
        self.df_benchmark['Daily Return'] = self.df_benchmark["Adj Close"].pct_change(1)
        # caculate metric index
        dr = self.df_position['Daily Return']
        br = self.df_benchmark['Daily Return']
        er = self.df_position['Daily Return'].mean()
        std = self.df_position['Daily Return'].std()
        var = self.df_position['Daily Return'].var()
        period = len(self.df_position.index)
        self.sharpe = (er - self.riskfree) / std
        self.sharpe = round(self.sharpe * np.sqrt(period) , 4)
        tmp_dr, tmp_br = dr.align(br, join='inner', axis=0)
        (self.beta, self.alpha) = stats.linregress(tmp_br[1:], tmp_dr[1:])[0:2]
        self.alpha = np.round(self.alpha , 4)
        self.beta = np.round(self.beta , 4)
        self.var = round(var , 4)
        self.treynor = round((er - self.riskfree) / self.beta , 4) # consider annualized * period
        diff = dr - br
        self.infomation = round(diff.mean() / diff.std(),4)  # consider annualized * np.sqrt(period)

    def portfolio(self):
        print('\n-----------------------------------Portfolio Report-----------------------------------')
        print('Portfolio', sorted(self.pairs))
        print('Inital Money: ', self.init_money, ' NTD')
        print('Number of Invest Stocks: ', len(self.stockList), ' stocks')
        print('Portfolio Start : ', self.start,
              '\nPortfolio End   : ', self.end)
        print('Invest Days: ', self.n_days, ' days',
              '\nTrading Days: ', len(self.df_position.index), ' days')
        print('Cumulative Return : {} %'.format(self.cum_return))
        print('Net Assets: {}'.format(
            round(self.df_position['Total Position'][-1]), 4), ' NTD')
        print('Risk Free Rate:', self.riskfree, '%')
        print('-----------------------------------Portfolio Report-----------------------------------\n')

    def metric(self,index = None):
        self.df_metric = pd.DataFrame()
        self.df_metric = pd.concat([self.df_metric, pd.DataFrame({'Sharpe Ratio':[self.sharpe],
                                                                 'BETA':[self.beta],
                                                                 'Variance':[self.var],
                                                                 'Treynor Ratio':[self.treynor],
                                                                 'Jensen Ratio':[self.alpha],
                                                                 'Information Ratio':[self.infomation]}
                                                                 )],axis=1)
        if index != None:
            self.df_metric.index = [index]
        return self.df_metric

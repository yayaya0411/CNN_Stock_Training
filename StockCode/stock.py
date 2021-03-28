#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import mpl_finance as mpf
import seaborn as sns
import datetime as datetime
import talib
from sklearn import preprocessing
# from StockCode.self_money import *

class dataFrame():
    #範例初始化 dataFrame('2017/1/1', '2017/12/31', '0050')
    def __init__(self, start, end, stockName):
        s = start.split('/')
        e = end.split('/')
        start = datetime.datetime(int(s[0]), int(s[1]), int(s[2]))
        end = datetime.datetime(int(e[0]), int(e[1]), int(e[2]))
        if stockName == '^TWII':
            self.df = pdr.DataReader(stockName, 'yahoo', start=start, end=end)
        else:
            self.df = pdr.DataReader(stockName+'.TW', 'yahoo', start=start, end=end)
        self.df.index = self.df.index.format(
            formatter=lambda x: x.strftime('%Y-%m-%d'))

    def printDF(self, num=0):
        if num==0:
            print(self.df)
        else:
            print(self.df[:num])

    def MaxMinNormalization(x, Max, Min):
        x = (x - Min) / (Max - Min)
        return x

    #對輸入的欄位做標準化並新增一個欄位“Norm”+"輸入"
    def Normalization(self, String):
        maxi = max(self.df[String])
        mini = min(self.df[String])
        nor = []
        for num in self.df[String]:
            nor.append((num-mini)/(maxi-mini))
        self.df['Norm'+String] = nor
    #畫出指定欄位的線圖

    def drawChoseColumn(self, String):
        fig, ax = plt.subplots()
        data = self.df[String]
        fig = plt.figure(figsize=(24, 20))
        ax = fig.add_axes([0, 0.3, 1, 0.4])
        ax.set_xticks(range(0, len(self.df.index), 10))
        ax.set_xticklabels(self.df.index[::10])
        ax.plot(data, label=String)
        plt.grid(True)
        plt.show()
    #畫出K線圖
    def drawKLine(self):
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        fig = plt.figure(figsize=(24, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(range(0, len(self.df.index), 10))
        ax.set_xticklabels(self.df.index[::10])
        mpf.candlestick2_ochl(ax, self.df['Open'], self.df['Close'], self.df['High'],
                              self.df['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
        plt.grid(True)
        plt.show()

    # 畫出MA_line，後面加上true的話會印出原先的陰陽線，若為false，則只印出MA線;num1,num2為日期長短
    def drawMA(self, det=False, num1=3, num2=10):
        fig, ax = plt.subplots()
        sma1 = talib.SMA(np.array(self.df['Close']), num1)
        sma2 = talib.SMA(np.array(self.df['Close']), num2)
        fig = plt.figure(figsize=(24, 20))
        ax = fig.add_axes([0, 0.3, 1, 0.4])  # 加上留白
        ax.set_xticks(range(0, len(self.df.index), 10))
        ax.set_xticklabels(self.df.index[::10])
        if det == True:
            mpf.candlestick2_ochl(ax, self.df['Open'], self.df['Close'], self.df['High'],
                self.df['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
        ax.plot(sma1, label=str(num1)+'_MA')
        ax.plot(sma2, label=str(num2)+'_MA')
        plt.grid(True)
        plt.show()
    #新增平均線欄位
    def calMA(self,  num):
        sma = talib.SMA(np.array(self.df['Close']), num)
        self.df[str(num)+'MA'] = np.nan_to_num(sma)
   #畫出交易量
    def drawVolume(self):
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(24, 20))
        ax = fig.add_axes([0, 0.3, 1, 0.4])
        mpf.volume_overlay(ax, self.df['Open'], self.df['Close'], self.df['Volume'],
                           colorup='r', colordown='g', width=0.5, alpha=0.8)
        ax.set_xticks(range(0, len(self.df.index), 10))
        ax.set_xticklabels(self.df.index[::10])
        plt.grid(True)
        plt.show()

    def drawKD(self):  # 畫ＫＤ圖
        fig, ax = plt.subplots()
        K, D = self.returnKD()
        fig = plt.figure(figsize=(24, 20))
        ax = fig.add_axes([0, 0.3, 1, 0.4])
        ax.set_xticks(range(0, len(self.df.index), 10))
        ax.set_xticklabels(self.df.index[::10])
        ax.plot(D, label='D_line')
        ax.plot(K, label='K_line')
        plt.grid(True)
        plt.show()

    def calKD(self):  # 計算KD
        self.df['K'], self.df['D'] = talib.STOCH(
            self.df['High'], self.df['Low'], self.df['Close'])
        self.df['K'] = np.nan_to_num(self.df['K'])
        self.df['D'] = np.nan_to_num(self.df['D'])
    def calRSI(self):
        self.df['RSI'] = talib.RSI(self.df['Close'])
    def calRSV(self, num):  # 計算RSV num為計算基準
        rsv = (self.df['Close']-self.df['Close'].rolling(window=num).min()) / \
            (self.df['Close'].rolling(window=num).max() -
             self.df['Close'].rolling(window=num).min())*100
        self.df['RSV'] = np.nan_to_num(rsv)
    def calBias(self, num):
        self.calMA(num)
        self.df['BIAS'] = (self.df['Close'] -
                           self.df[str(num)+'MA'])/self.df[str(num)+'MA']
        where_are_inf = np.isinf(self.df['BIAS'])
        self.df['BIAS'][where_are_inf]=0
    def returnRSV(self):
        return self.df['RSV']

    def returnKD(self):  # 回傳指標數值（list）
        return self.df['K'], self.df['D']

    def returnMA(self, num):  # 回傳 MA (list)
        return talib.SMA(np.array(self.df['Close']), num)

    def TradeByChoice(self):
        choice = str(input("請輸入 KD 或是 MA 或是 volume 或是 BIAS 或是 RSI+MA："))
        money = int(input("請輸入初始資產："))
        myM = stockEvn_single(money)
        re = pd.DataFrame()
        if choice == "MA":
            num1 = int(input("請輸入 MA慢線週期:"))
            self.calMA(num1)
            num2 = int(input("請輸入 MA快線週期:"))
            self.calMA(num2)
            for i in range(1, len(self.df[str(num1)+'MA'])):
                nowSlow = self.df[str(num1)+'MA'][i]
                nowFast = self.df[str(num2)+'MA'][i]
                lastSlow = self.df[str(num1)+'MA'][i-1]
                lastFast = self.df[str(num2)+'MA'][i-1]
                ordertime = self.df.index[i]
                orderprice = self.df['Close'][i]
                if lastFast <= lastSlow and nowFast > nowSlow:  # 黃金交叉
                    data = {'action': 1, 'time': ordertime, 'price': orderprice}
                    myM.doAction(orderprice, 1, inAll=True)
                    myM.getInfo()
                elif lastFast >= lastSlow and nowFast < nowSlow:  # 死亡交叉
                    data = {'action': 2, 'time': ordertime, 'price': orderprice}
                    myM.doAction(orderprice, 2, outAll=True)
                    myM.getInfo()
                else:
                    data = {'action': 0, 'time': ordertime, 'price': orderprice}
                    myM.doAction(orderprice, 0, inAll=True)
                    #myM.getInfo()
                re = re.append(data, ignore_index=True)
            return re
        elif choice == "KD":
            self.calKD()
            k, d = self.df['K'], self.df['D']
            for i in range(1, len(k)):
                thisK = k[i]
                thisD = d[i]
                LastK = k[i-1]
                LastD = d[i-1]
                ordertime = self.df.index[i]
                orderprice = self.df['Close'][i]
                if LastK <= LastD and thisK > thisD:  # 黃金交叉
                    data = {'action': 1, 'time': ordertime, 'price': orderprice}
                    myM.doAction(orderprice, 1, inAll=True)
                    myM.getInfo()
                elif LastK >= LastD and thisK < thisD:  # 死亡交叉
                    data = {'action': 2, 'time': ordertime, 'price': orderprice}
                    myM.doAction(orderprice, 2, outAll=True)
                    myM.getInfo()
                else:
                    data = {'action': 0, 'time': ordertime, 'price': orderprice}
                    myM.doAction(orderprice, 0, inAll=True)
                    #myM.getInfo()
                re = re.append(data, ignore_index=True)
            return re
        elif choice == 'volume':
            num = int(input("數量平均週期"))
            for i in range(num, len(self.df['Volume'])):
                ordertime = self.df.index[i]
                orderprice = self.df['Close'][i]
                if self.df['Volume'][i]* num > sum( self.df['Volume'][i-num:i])*(1.1):
                    data = {'action': 1, 'time': ordertime,'price': orderprice}
                    myM.doAction(orderprice, 1, inAll=True)
                    myM.getInfo()
                elif self.df['Volume'][i] * num < sum(self.df['Volume'][i-num:i])*(0.9):
                    data = {'action': 2, 'time': ordertime,
                         'price': orderprice}
                    myM.doAction(orderprice, 2, outAll=True)
                    myM.getInfo()
                else:
                    data = {'action': 0, 'time': ordertime,
                         'price': orderprice}
                    myM.doAction(orderprice, 0, inAll=True)
                    #myM.getInfo()
                re = re.append(data, ignore_index=True)
            return re
        elif str.lower(choice) == str.lower('BIAS'):
            num = int(input("乖離率計算週期(一般為6)"))
            self.calBias(num)
            for i in range(num, len(self.df['BIAS'])):
                ordertime = self.df.index[i]
                orderprice = self.df['Close'][i]
                if self.df['BIAS'][i]<-0.002: #乖離率過低 進場
                    data = {'action': 1, 'time': ordertime,'price': orderprice}
                    myM.doAction(orderprice, 1, inAll=True)
                    myM.getInfo()
                elif self.df['BIAS'][i] > 0.001: #乖離率過大 出場
                    data = {'action': 2, 'time': ordertime,
                         'price': orderprice}
                    myM.doAction(orderprice, 2, outAll=True)
                    myM.getInfo()
                else:
                    data = {'action': 0, 'time': ordertime,
                         'price': orderprice}
                    myM.doAction(orderprice, 0, inAll=True)
                    #myM.getInfo()
                re = re.append(data, ignore_index=True)
            return re
        elif str.lower(choice) == str.lower('RSI+MA'):
            num1 = int(input("請輸入 MA慢線週期:"))
            self.calMA(num1)
            num2 = int(input("請輸入 MA快線週期:"))
            self.calMA(num2)
            self.calRSI()
            for i in range(1, len(self.df[str(num1)+'MA'])):
                nowSlow = self.df[str(num1)+'MA'][i]
                nowFast = self.df[str(num2)+'MA'][i]
                lastSlow = self.df[str(num1)+'MA'][i-1]
                lastFast = self.df[str(num2)+'MA'][i-1]
                ordertime = self.df.index[i]
                orderprice = self.df['Close'][i]
                if self.df['RSI'][i]>50 and lastFast <= lastSlow and nowFast > nowSlow:  # 黃金交叉
                    data = {'action': 1, 'time': ordertime,
                            'price': orderprice}
                    myM.doAction(orderprice, 1, inAll=True)
                    myM.getInfo()
                elif self.df['RSI'][i] <50 and lastFast >= lastSlow and nowFast < nowSlow:  # 死亡交叉
                    data = {'action': 2, 'time': ordertime,
                            'price': orderprice}
                    myM.doAction(orderprice, 2, outAll=True)
                    myM.getInfo()
                else:
                    data = {'action': 0, 'time': ordertime,
                            'price': orderprice}
                    myM.doAction(orderprice, 0, inAll=True)
                    #myM.getInfo()
                re = re.append(data, ignore_index=True)
            return re

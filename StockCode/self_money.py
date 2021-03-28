#-*- coding: utf-8 -*-
#已加入手續費
class stockEvn_single:
    stockAmount = 0
    stockLastPrice = 0

    def __init__(self, money=1000):
        self.leftMoney = money
        self.initMoney = money

    def getInfo(self):
        print('stock count : ', self.stockAmount,
              ', stock value : ', self.stockLastPrice,
              ', money leave : ', self.leftMoney,
              ',initial money:', self.initMoney,
              ' now_assets: ', self.calAssets(),
              'win_percent: ', self.winPercent())
        print("=========================================")
    def winPercent(self):
        percent = (self.calAssets()- self.initMoney)*100/self.initMoney
        string = str(percent)
        string += "%"
        return string
    def calAssets(self):
        return int(self.stockAmount * self.stockLastPrice * 1000 + self.leftMoney)

    #可輸入六個變數：
    #現在股價 動作 想買入的錢 想賣出的股票數 是否全部買進 是否全部賣出
    def doAction(self, stock_price, action, inMoney=0, sellStockNum=0, inAll=False, outAll=False):
        # action = 0: do nothing; 1: buy; 2: sell
        if action == 0:
            self.stockLastPrice = stock_price
            #print("Do nothing")
            pass
        if action == 1:
            if inAll == True:
                limitNum = int(self.leftMoney / (stock_price * 1000*(100+0.1425)/100))
                self.stockAmount += limitNum
                self.leftMoney -= int(limitNum *
                                      stock_price * 1000*(100+0.1425)/100)
                print("All in")
            elif inMoney > self.leftMoney:
                print("你的資產不足")
            else:
                limitNum = int(inMoney / (stock_price * 1000*(100+0.1425)/100))
                self.stockAmount += limitNum
                self.leftMoney -= int(limitNum *
                                      stock_price*100*(100+0.1425)/100)
                print("交易完成")
            self.stockLastPrice = stock_price
        if action == 2:
            if outAll == True and self.stockAmount > 0:
                self.leftMoney += int(self.stockAmount *
                                      stock_price * 1000*(100-0.4425)/100)
                self.stockAmount = 0
                print("全數賣出")
            elif sellStockNum < self.stockAmount:
                print("沒這麼多股票能賣出")
            else:
                self.leftMoney += int(sellStockNum *
                                      stock_price * 1000*(100-0.4425)/100)
                self.stockAmount -= sellStockNum
                print("交易完成")
            self.stockLastPrice = stock_price

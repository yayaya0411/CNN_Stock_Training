
### StockLake

建立一個股票池，並設定這個池子股價的起迄日
```py
# 股票List
lstocklist = ['0050','2888','2884','2882','2887','2886']
# 建立StockLake class
lake = StockLake(start_date = '2010/01/01', end_date = '2010/12/31')
# 將這個池子的股票加入StockLake
lake.add(lstocklist)
# 讀取股價資料
lake.load_stock() # 預設benchmark為台股加權指數 ^TWII
```

使用`download`下載 stocklake csv
預設下載至`StockLake/Stock`, `StockLake/Benchmark`
```py
# default 下載至 StockLake/StockLake_datetime.csv
lake.download()
# 可指定名稱 download to StockLake/aaa.csv
lake.download(name = 'aaa.csv')
```
使用`load`讀取資料夾csv
```py
lake = StockLake(start_date = '2010/01/01', end_date = '2010/12/31')
# 預設讀取SockLake資料夾內以名稱排序第一筆的csv
lake.load()
# 可指定讀取檔案路徑
lake.load(file = path)
```

### Portfolio

建立一個投資組合，並選擇這個投資組合要使用的股票池，投資組合的期間需要在股票池的起迄日內
```py
test = Portfolio(start = '2010/12/01', end = '2010/12/31')
```

使用`add` 以List輸入股票代碼，可以將該股票加入投資組合，預設資金分配比率為均分
```py
test.add(['0050','2412','2330','2888'])
```
使用`assignment`以List指定股票分配比率，會配合股票代碼的順序指定分配比率
若`assignment` 比率加總 !=1，則對每一個element / sum(assignment_list) 做標準化處理
```py
test.assignment([0.1,0.2,0.3,0.4])
```
使用`now_stockList`確認股票及其資金分配比率
```py
test.now_stockList()
```
若要打算測試投資組合，刪除部分股票，<br/>
使用`remove`  以List 輸入股票代碼，可將股票移出投資組合 <br/>
使用`refresh` 清空投資組合股票
```py
test.remove(['2888'])
test.refresh()
```

讀取股價資料及計算資訊，並選擇要使用的股票池
```py
test.load_data(StockLake = lake)
```

觀察投資組合結果
```py
test.portfolio()
```

觀察投資組合指標
```py
test.metric()
```

使用以下方法，若想確認投資組合相關資料
```py
test.df  # 股價原始資料
test.df_benchmark # Benchmark 資料
test.df_position # 投資組合資金水位資料
test.var # Variance
test.beta # BETA
test.alpha # Jensen Ratio
test.sharpe # Sharpe Ratio
test.treynor # Treynor Ratio
test.infomation # Information Ratio
```

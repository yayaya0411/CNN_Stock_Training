from StockCode.Portfolio import *
from tensorflow.keras.utils import to_categorical

def call_portfolio(stocklist):
    lake = StockLake(start_date = '2000/01/01', end_date = '2020/07/01')
    lake.add(stocklist) #選擇池子的股票
    # download to csv
    # lake.load_stock()
    # lake.download()
    # lake.load()

    # load first file in stocklake folder
    stock_path = 'StockLake/Stock/0050.csv'
    benchmark_path = 'StockLake/Benchmark/0050_benchmark.csv'
    lake.load(file = stock_path,benchmark = benchmark_path)
    # stock_path = 'StockLake/Stock/0050_portfolio.csv'
    # benchmark_path = 'StockLake/Benchmark/0050_portfolio_benchmark.csv'
    # lake.load(file = stock_path,benchmark = benchmark_path)
    return lake

def portfolio_data(lake, stocklist):
    Data = Portfolio(start = '2003/01/01', end = '2020/06/30')
    Data.add(stocklist)
    Data.load_data(StockLake = lake)
    Data.portfolio()
    return Data.df

## make label
def add_label(df, ratio_gap ,stocklist ,types = 'next', n = 1):
    tmp_df = pd.DataFrame()
    if 'Label' in df.columns:
        df.drop(['Label'], axis=1, inplace = True)
    if types == 'next':
        print('Class Label up/down ratio gap {} %'.format(ratio_gap))
        for stock in stocklist:
            df_ = df[df['Stock']==stock]
            df_['Label_r']=((df_['Close'] / df_['Close'].shift(n))-1)*100
            df_ = df_.fillna(0)
            df_['Label'] = 0
            df_ = df_.fillna(0)
            df_.loc[round(df_['Label_r'],2) < -ratio_gap, 'Label'] = 1 # fall over 1% ratio
            df_.loc[round(df_['Label_r'],2) > ratio_gap, 'Label'] = 2 # raise over 1% ratio
            df_.loc[(df_['Label'] != 1) & (df_['Label'] != 2) , 'Label'] = 0 # percent not Significant change
            df_['Label'] = df_['Label'].astype(int).astype('O')
            df_.drop(['Label_r'],axis=1,inplace=True)
            tmp_df = tmp_df.append(df_)
    if types == 'test_2class':
        print('test 2 class for (0,1) which 1 is rise {} % 0 is else'.format(ratio_gap))
        for stock in stocklist:
            df_ = df[df['Stock']==stock]
            df_['Label_r']=((df_['Close'] / df_['Close'].shift(n))-1)*100
            df_ = df_.fillna(0)
            df_['Label'] = 0
            df_ = df_.fillna(0)
            df_.loc[round(df_['Label_r'],2) > ratio_gap, 'Label'] = 1 # raise over 1% ratio
            df_.loc[df_['Label'] != 1 , 'Label'] = 0 # percent not Significant change
            df_['Label'] = df_['Label'].astype(int).astype('O')
            df_.drop(['Label_r'], axis = 1, inplace = True)
            tmp_df = tmp_df.append(df_)
    if types == 'slope':
        print('Slope Label with regression ')
        for stock in stocklist:
            df_ = df[df['Stock']==stock]

    return tmp_df

## Standardization Function
def standar(df,y):
    norm = df.iloc[:,:-1].apply(lambda x: (x - np.mean(x)) / np.std(x))
    norm_df = pd.concat([norm,df[y]],axis=1)
    return norm_df

## Caculate n Days SMA
def sma(df,n,y):
    ma  = df.iloc[:,:-1].rolling(n).mean()
    ma_df = pd.concat([ma,df[y]],axis=1)
    ma_df = ma_df.iloc[n-1:,:]
    return ma_df

## Caculate n Days EMA
def ema(df,n,y):
    ma  = df.iloc[:,:-1].ewm(span=n).mean()
    ma_df = pd.concat([ma,df[y]],axis=1)
    ma_df = ma_df.iloc[n-1:,:]
    return ma_df

## Change Data to Ratio
def to_ratio(df,y):
    tmp = df.iloc[:,:-1].apply(lambda x : ((x / x.shift(1))-1)*100)
    tmp_df = pd.concat([tmp,df[y]],axis=1)
    tmp_df = tmp_df.iloc[1:,:]
    return tmp_df

## align data shape
def align_shape(df_1, df_2):
    if df_1.shape[0] > df_2.shape[0]:
        df_1 = df_1.iloc[-df_2.shape[0]:,:]
    else:
        df_2 = df_2.iloc[-df_1.shape[0]:,:]
    return df_1, df_2

## split data to train valid test
def split_data(df,split_date):
    start = split_date[0]
    if len(split_date)==3:
        mid = split_date[1]
    end = split_date[-1]
    train = df[(df.index >= start ) & (df.index < mid)]
    valid = df[(df.index >= mid ) & (df.index < end)]
    test = df[(df.index >= end)]
    return train,valid,test

## specific label data n times increase
def increase_data(X, y, label, n=1):
    mask = np.argmax(y,axis=1) == label
    tmp_X = np.tile(X[mask],(n,1,1))
    tmp_y = np.tile(y[mask],(n,1))
    ori = X.shape[0]
    increase = tmp_X.shape[0]
    X = np.concatenate((X,tmp_X))
    y = np.concatenate((y,tmp_y))
    print('Label {} Data increase {} from {} to {}'.format(label, increase, ori,increase+ori))
    return X, y

## constructe training input
def train_windows(df, label, window, predict_day=1):
    X, y = [], []
    for i in range(df.shape[0]-predict_day-window):
        X.append(np.array(df.iloc[i : i + window, : -1]))
        y.append(np.array(df.iloc[i + window : i + window + predict_day][label]))
    return np.array(X), np.array(y)

## transform to input
def input_output(train, valid, test, y, window):
    X_train, Y_train  = train_windows(train, label = y, window = window)
    X_valid, Y_valid  = train_windows(valid, label = y, window = window)
    X_test, Y_test  = train_windows(test, label = y, window = window)
    y_train = to_categorical(Y_train)
    y_valid = to_categorical(Y_valid)
    y_test = to_categorical(Y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def np_count(array):
    count_array = np.argmax(array,axis=1)
    count = pd.Series(count_array)
    return count

import baostock as bs
import pandas as pd
import time

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

# 获取所有证券的基本资料
# type 证券类型，其中1：股票，2：指数，3：其它，4：可转债，5：ETF
# status 上市状态，其中1：上市，0：退市
rs = bs.query_stock_basic()
print('query_stock_basic respond error_code:' + rs.error_code)
print('query_stock_basic respond  error_msg:' + rs.error_msg)

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv("data/stock_basic.csv", encoding="gbk", index=False)

# 筛选出type为1（股票）的证券
selected_stocks = result[(result['type'] == '1')]

period_begin = '2013-01-01'
period_end = '2024-12-31'

# 循环调用获取沪深A股日频数据
for index, row in selected_stocks.iterrows():
    stock_code = row['code']
    stock_name = row['code_name']
    rs = bs.query_history_k_data_plus(stock_code,
                                      "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                                      start_date=period_begin, end_date=period_end,
                                      frequency="d", adjustflag="1")  # 复权类型，默认不复权：3；1：后复权；2：前复权。

    if rs.error_code == '0':
        result_list = []
        while rs.next():
            # 获取一条记录，将记录合并在一起
            result_list.append(rs.get_row_data())

        if len(result_list) > 0:
            # 将结果转换为DataFrame
            result = pd.DataFrame(result_list, columns=rs.fields)

            # 结果集输出到csv文件
            filename = f"data/history/history_A_{stock_code}.csv"
            result.to_csv(filename, encoding="gbk", index=False)
            print(f"processed {stock_code}")
        else:
            print(f'股票{stock_code}：{stock_name}在区间里没有日线数据')
    else:
        print(f'股票{stock_name}获取日线错误：{rs.error_msg}')

    # 暂停100毫秒
    time.sleep(0.1)

#### 登出系统 ####
bs.logout()

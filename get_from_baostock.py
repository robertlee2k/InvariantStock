import os
import time
from concurrent.futures import ThreadPoolExecutor

import baostock as bs
import numpy as np
import pandas as pd
from tqdm import tqdm


def norm(df):
    mean = df.mean()
    std = df.std()
    df = (df - mean) / std
    return df


def multi_normalize(df_list, num_threads=4):
    # 创建线程池
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务并获取未来对象
        futures = [executor.submit(norm, df) for _, df in df_list]

        # 获取结果
        results = [future.result() for future in futures]

    # 将结果重新组合成一个数据框
    normalized_df = pd.concat(results)
    return normalized_df


class StockDataFetcher:
    def __init__(self):
        self.period_begin = '2013-01-01'
        self.period_end = '2024-12-31'

    @staticmethod
    def login():
        lg = bs.login()
        print('login respond error_code:' + lg.error_code)
        print('login respond  error_msg:' + lg.error_msg)

    @staticmethod
    def logout():
        bs.logout()

    @staticmethod
    def fetch_stock_basic(stock_basic_csv):
        if not os.path.exists(stock_basic_csv):
            rs = bs.query_stock_basic()
            print('query_stock_basic respond error_code:' + rs.error_code)
            print('query_stock_basic respond  error_msg:' + rs.error_msg)

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=rs.fields)
            result.to_csv(stock_basic_csv, encoding="gbk", index=False)
        else:
            result = pd.read_csv(stock_basic_csv, encoding='gbk')

        selected_stocks = result[(result['type'] == '1')]
        return selected_stocks

    def fetch_history_k_data(self, target_stock_codes, stock_history_dir):
        if not os.path.exists(stock_history_dir):
            os.makedirs(stock_history_dir)

        for index, row in target_stock_codes.iterrows():
            stock_code = row['code']
            stock_name = row['code_name']
            filename = f"data/history/history_A_{stock_code}.csv"

            if os.path.exists(filename):
                print(f"文件 {filename} 已存在，跳过 {stock_code}")
                continue

            rs = bs.query_history_k_data_plus(stock_code,
                                              "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                                              start_date=self.period_begin, end_date=self.period_end,
                                              frequency="d", adjustflag="1")

            if rs.error_code == '0':
                result_list = []
                while rs.next():
                    result_list.append(rs.get_row_data())

                if len(result_list) > 0:
                    result = pd.DataFrame(result_list, columns=rs.fields)
                    result.to_csv(filename, encoding="gbk", index=False)
                    print(f"processed {stock_code}")
                else:
                    print(f'股票{stock_code}：{stock_name}在区间里没有日线数据')
            else:
                print(f'股票{stock_name}获取日线错误：{rs.error_msg}')

            time.sleep(0.1)

    @staticmethod
    def convert_to_float_or_zero(series):
        # 预处理空值和空白字符串
        series = series.fillna(0).replace(r'^\s*$', 0, regex=True)
        # 使用 pd.to_numeric 进行转换，errors='coerce' 会将无法转换的值设为 NaN
        series = pd.to_numeric(series, errors='coerce')
        # 将 NaN 替换为 0
        series = series.fillna(0)
        return series

        # if pd.isna(x) or str(x).strip() == "":
        #     return 0
        # try:
        #     return float(x)
        # except ValueError:
        #     return 0

    @staticmethod
    def read_csv_file(file_path):
        return pd.read_csv(file_path, encoding='gbk')

    @staticmethod
    def prepare_dataset(history_dir, pickle_file, need_norm=True):
        # 从data/history/目录读取所有的CSV文件
        csv_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.csv')]

        # 使用线程池并行读取文件
        with ThreadPoolExecutor() as executor:
            # 使用 tqdm 包装 executor.map 以显示进度条
            dfs = list(tqdm(executor.map(StockDataFetcher.read_csv_file, csv_files), total=len(csv_files),
                            desc="读取文件进度"))

        # 合并所有数据框
        full_dataset = pd.concat(dfs, ignore_index=True)

        print("处理turn的特殊值.....")

        # 使用 apply 方法进行向量化操作
        full_dataset['turn'] = StockDataFetcher.convert_to_float_or_zero(full_dataset['turn'])

        print("根据原始数据计算流通市值和流通股本.....")
        # 根据原始数据计算“流通市值”和“流通股本”
        # 如果'turn’ 不为零，流通股本 = volumn/turn
        # 流通市值 = 流通股本* close
        # full_dataset['Circulated Shares'] = full_dataset.apply(
        #    lambda row: row['volume'] / row['turn'] if row['turn'] != 0 else None, axis=1)
        # 对于大规模的df，使用 pandas 的 where 方法 会比apply更快
        full_dataset['Circulated Shares'] = (full_dataset['volume'] / full_dataset['turn']).where(
            full_dataset['turn'] != 0, np.nan)
        full_dataset['Circulated Market Value'] = full_dataset['Circulated Shares'] * full_dataset['close']

        print("转换日期为Timestamp类型.....")
        # 将 date 列转换为 Timestamp 类型
        full_dataset['date'] = pd.to_datetime(full_dataset['date'])

        print("把dataset按日期升序排序....")
        # 把dataset按日期、code升序排序
        full_dataset = full_dataset.sort_values(by='date', ascending=True)

        print("添加label字段.....")
        # 添加'label'字段
        # TODO 虽然pctChg一般是-10到10之间的数值，但这里应该还是要做normalization比较好
        full_dataset['label'] = full_dataset.groupby('code')['pctChg'].shift(-1)

        # 筛选出将要被前向填充的行
        rows_to_fill = full_dataset[full_dataset.isna().any(axis=1)]
        print(f"使用前向填充的数据行有{len(rows_to_fill)}行")
        # # 保存将要被前向填充的行到CSV文件
        # rows_to_fill.to_csv('data/rows_to_fill_na.csv', index=False)

        # 使用groupby和ffill填充缺失值，并保留原来的'code'列
        # 排除 'date' 和 'code' 列
        columns_to_apply = full_dataset.drop(columns=['date', 'code']).columns
        # 使用 groupby 和 apply
        full_dataset[columns_to_apply] = full_dataset.groupby('code')[columns_to_apply].apply(
            lambda group: group.ffill()).reset_index(
            drop=True)

        # 检查数据集中是否仍有NA值
        if full_dataset.isna().any().any():
            print("数据集中有以下行包含NA值：")
            print(full_dataset[full_dataset.isna().any(axis=1)])

            # 删除包含NA值的行
            full_dataset.dropna(inplace=True)
            print("已删除包含NA值的行")

        print("把dataset重新按日期、code升序排序....")
        # 把dataset按日期、code升序排序
        full_dataset = full_dataset.sort_values(by=['date', 'code'], ascending=True)

        print("在清理好的数据上，设置复合索引.....")
        # 设置索引
        full_dataset.set_index(['date', 'code'], inplace=True)

        if need_norm:
            # 除label字段之外全部做归一化
            full_dataset[full_dataset.columns.drop("label")] = multi_normalize(
                [*full_dataset[full_dataset.columns.drop("label")].groupby("date")])

        full_dataset.to_pickle(pickle_file)
        print("保存的结果数据集sample----------")
        print(full_dataset)


def main():
    fetcher = StockDataFetcher()
    fetcher.login()

    stock_basic_csv = "data/stock_basic.csv"
    selected_stocks = fetcher.fetch_stock_basic(stock_basic_csv)

    stock_history_dir = "data/history"
    if not os.path.exists(stock_history_dir):
        fetcher.fetch_history_k_data(selected_stocks, stock_history_dir)
    fetcher.logout()

    pickle_file = "data/adataset-norm.pkl"
    if not os.path.exists(pickle_file):
        fetcher.prepare_dataset(stock_history_dir, pickle_file, need_norm=True)


if __name__ == "__main__":
    main()

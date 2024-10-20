import os
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Pool, Manager
from tqdm import tqdm
from functools import partial

# Define argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the dataset for training, validation, and testing.")
    parser.add_argument('--data_dir', type=str, default='./data', help='directory containing the data')
    parser.add_argument('--train_date', type=str, default='2022-12-31', help='end date for training data')
    parser.add_argument('--valid_date', type=str, default='2023-12-31', help='end date for validation data')
    parser.add_argument('--test_date', type=str, default='2024-12-31', help='end date for test data')
    parser.add_argument('--seq_len', type=int, default=20, help='sequence length for processing')
    return parser.parse_args()

def norm(df_tuple):

    df = df_tuple[1]
    mean = df.mean()
    std = df.std()
    df = (df-mean)/std
    return df
def multi_normalize(df_list):
    pool = multiprocessing.Pool()
    results = pool.map(norm, df_list)
    df = pd.concat(results)
    pool.close()
    pool.join()
    return df

if __name__ == '__main__':

    # 处理多种空值情况
    def convert_to_float_or_zero(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0
        try:
            return float(x)
        except ValueError:
            return 0

    args = parse_arguments()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    pickle_file = os.path.join(args.data_dir, "adataset_norm.pkl")

    if os.path.exists(pickle_file):
        # 直接加载Pickle文件
        dataset = pd.read_pickle(pickle_file)
    else:
        # 从data/history/目录读取所有的CSV文件
        history_dir = os.path.join(args.data_dir, 'history')
        csv_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.csv')]

        # 读取并拼接所有CSV文件
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file, encoding='gbk')
            dfs.append(df)
        dataset = pd.concat(dfs)

        # 使用 apply 方法进行向量化操作
        dataset['turn'] = dataset['turn'].apply(convert_to_float_or_zero)

        # 根据原始数据计算“流通市值”和“流通股本”
        # 如果'turn’ 不为零，流通股本 = volumn/turn
        # 流通市值 = 流通股本* close
        dataset['Circulated Shares'] = dataset.apply(
            lambda row: row['volume'] / row['turn'] if row['turn'] != 0 else None, axis=1)
        dataset['Circulated Market Value'] = dataset['Circulated Shares'] * dataset['close']

        # 设置索引
        dataset.set_index(['date', 'code'], inplace=True)
        # 把dataset按日期升序排序
        dataset = dataset.sort_values(by='date', ascending=True)
        # 添加'label'字段
        dataset['label'] = dataset.groupby('code')['pctChg'].shift(-1)

        # 筛选出将要被前向填充的行
        rows_to_fill = dataset[dataset.isna().any(axis=1)]
        # # 保存将要被前向填充的行到CSV文件
        # rows_to_fill.to_csv('data/rows_to_fill_na.csv', index=False)
        # 处理缺失值
        dataset = dataset.groupby('code').ffill()

        # 检查数据集中是否仍有NA值
        if dataset.isna().any().any():
            print("数据集中有以下行包含NA值：")
            print(dataset[dataset.isna().any(axis=1)])

            # 删除包含NA值的行
            dataset.dropna(inplace=True)
            print("已删除包含NA值的行")

        # Convert datetime index to Timestamp
        dataset.index = dataset.index.set_levels(pd.to_datetime(dataset.index.levels[0]), level=0)

        dataset[dataset.columns.drop("label")] = multi_normalize(
            [*dataset[dataset.columns.drop("label")].groupby("date")])

        dataset.to_pickle(pickle_file)

    print(dataset)

    train_date = pd.to_datetime(args.train_date)
    valid_date = pd.to_datetime(args.valid_date)
    test_date = pd.to_datetime(args.test_date)

    # 切分索引
    train_index = dataset.index[dataset.index.get_level_values("date") <= train_date]
    valid_index = dataset.index[(dataset.index.get_level_values("date") > train_date) & (dataset.index.get_level_values("date") <= valid_date)]
    test_index = dataset.index[dataset.index.get_level_values("date") > valid_date]
    print(test_index)

    # 将索引转换为 NumPy 数组并保存
    np.save(os.path.join(args.data_dir, "train_index.npy"), np.squeeze(train_index))
    np.save(os.path.join(args.data_dir, "valid_index.npy"), np.squeeze(valid_index))
    np.save(os.path.join(args.data_dir, "test_index.npy"), np.squeeze(test_index))

    print("Success!")


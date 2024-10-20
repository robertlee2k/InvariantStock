import os
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Pool, Manager

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

def get_index(dataset, date_list, index):
    sequence_length = 20
    date, stock = dataset.index[index]
    if date>date_list[-sequence_length]:
        return None
    date_seq = range(date_list.index(date),date_list.index(date)+ sequence_length)
    idx_list = [(date_list[i],stock) for i in date_seq]
    if not all(i in dataset.index for i in idx_list):
        return None

    return np.stack([dataset.index.get_indexer(idx_list)])

def multi_get_index(dataset, date_list, index_list):
    # pool = multiprocessing.Pool()
    # results = pool.starmap(get_index, [(dataset, date_list, index) for index in index_list])
    # pool.close()
    # pool.join()
    # results = [i for i in results if i is not None]
    # return np.stack(results)
    with Manager() as manager:
        shared_dataset = manager.list(dataset)  # 创建共享数据集
        with Pool() as pool:
            results = pool.starmap(get_index, [(shared_dataset, date_list, index) for index in index_list])
        return results

if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # dataset = pd.read_pickle(os.path.join(args.data_dir, "usdataset.pkl"))
    # dataset.set_index(["datetime", "instrument"], inplace=True)

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
        print(dataset)

        # 设置索引
        dataset.set_index(['date', 'code'], inplace=True)

        # 添加'label'字段
        dataset['label'] = dataset.groupby('code')['pctChg'].shift(-1)
        dataset.dropna(subset=['label'], inplace=True)

        # Convert datetime index to Timestamp
        dataset.index = dataset.index.set_levels(pd.to_datetime(dataset.index.levels[0]), level=0)

        dataset[dataset.columns.drop("label")] = multi_normalize(
            [*dataset[dataset.columns.drop("label")].groupby("date")])

        train_date = pd.to_datetime(args.train_date)
        valid_date = pd.to_datetime(args.valid_date)
        test_date = pd.to_datetime(args.test_date)

        train_range = range(0, len(dataset.loc[dataset.index.get_level_values("date") <= train_date]))
        valid_range = range(len(dataset.loc[dataset.index.get_level_values("date") <= train_date]),
                            len(dataset.loc[dataset.index.get_level_values("date") <= valid_date]))
        test_range = range(len(dataset.loc[dataset.index.get_level_values("date") <= valid_date]),
                           len(dataset))

        dataset.to_pickle(os.path.join(args.data_dir, pickle_file))

    date_list = list(dataset.index.get_level_values("date").unique())

    train_index = multi_get_index(dataset, date_list, [i for i in train_range])
    np.save(os.path.join(args.data_dir, "train_index.npy"), np.squeeze(train_index))
    valid_index = multi_get_index(dataset, date_list, [i for i in valid_range])
    np.save(os.path.join(args.data_dir, "valid_index.npy"), np.squeeze(valid_index))
    test_index = multi_get_index(dataset, date_list, [i for i in test_range])
    np.save(os.path.join(args.data_dir, "test_index.npy"), np.squeeze(test_index))

    print("Success!")


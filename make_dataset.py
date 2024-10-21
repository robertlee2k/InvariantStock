import os
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_index(index,dataset,date_list):
    try:
        print(f"getting training index for {index}")
        sequence_length = 20
        date, stock = dataset.index[index]
        if date > date_list[-sequence_length]:
            return None
        date_seq = range(date_list.index(date), date_list.index(date) + sequence_length)
        idx_list = [(date_list[i], stock) for i in date_seq]
        if not all(i in dataset.index for i in idx_list):
            return None
        return np.stack([dataset.index.get_indexer(idx_list)])
    except Exception as e:
        print(f"Error in get_index for index {index}: {e}")
        return None


def multi_get_index(index_list, dataset,date_list, num_threads=4):
    try:
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            print("Submitting tasks...")
            # 提交任务并获取未来对象
            futures = {executor.submit(get_index, index,dataset,date_list): index for index in index_list}
            print("Waiting tasks complete...")
            # 获取结果并显示进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing indices"):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error in future for index {futures[future]}: {e}")

        # 将结果堆叠成一个数组
        return np.stack(results)
    except Exception as e:
        print(f"Error in multi_get_index: {e}")
        return np.array([])

# Define argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the dataset for training, validation, and testing.")
    parser.add_argument('--data_dir', type=str, default='./data', help='directory containing the data')
    parser.add_argument('--train_date', type=str, default='2022-12-31', help='end date for training data')
    parser.add_argument('--valid_date', type=str, default='2023-12-31', help='end date for validation data')
    parser.add_argument('--test_date', type=str, default='2024-12-31', help='end date for test data')
    parser.add_argument('--seq_len', type=int, default=20, help='sequence length for processing')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    pickle_file = os.path.join(args.data_dir, "adataset-norm.pkl")

    # 直接加载Pickle文件
    dataset = pd.read_pickle(pickle_file)

    print(dataset)

    train_date = pd.to_datetime(args.train_date)
    valid_date = pd.to_datetime(args.valid_date)
    test_date = pd.to_datetime(args.test_date)

    train_range = range(0, len(dataset.loc[dataset.index.get_level_values("date") <= train_date]))
    valid_range = range(len(dataset.loc[dataset.index.get_level_values("date") <= train_date]),
                        len(dataset.loc[dataset.index.get_level_values("date") <= valid_date]))
    test_range = range(len(dataset.loc[dataset.index.get_level_values("date") <= valid_date]),
                       len(dataset))

    date_list = list(dataset.index.get_level_values("date").unique())

    train_index = multi_get_index([i for i in train_range],dataset,date_list)
    np.save(os.path.join(args.data_dir, "train_index.npy"), np.squeeze(train_index))
    valid_index = multi_get_index([i for i in valid_range],dataset,date_list)
    np.save(os.path.join(args.data_dir, "valid_index.npy"), np.squeeze(valid_index))
    test_index = multi_get_index([i for i in test_range],dataset,date_list)
    np.save(os.path.join(args.data_dir, "test_index.npy"), np.squeeze(test_index))

    print("Success!")


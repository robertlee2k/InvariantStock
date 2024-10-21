import os
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_index(index,dataset,date_list):
    try:
        # print(f"getting training index for {index}")
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

# 预处理数据结构
def preprocess_data(dataset, date_list):
    dataset_index_set = set(dataset.index)  # 使用集合加速查找
    date_to_index = {date: idx for idx, date in enumerate(date_list)}  # 建立日期到索引的映射
    dataset_index = pd.Index(dataset.index)  # 使用 pandas Index 类
    return dataset_index_set, date_to_index, dataset_index

# 优化后的 get_index_batch 函数
def get_index_batch(batch, dataset_index_set, date_to_index, dataset_index, date_list):
    try:
        sequence_length = 20
        results = []

        for index in batch:
            try:
                date, stock = dataset_index[index]  # 直接从 dataset_index 获取索引信息
                if date > date_list[-sequence_length]:
                    continue

                start_idx = date_to_index[date]
                end_idx = start_idx + sequence_length
                if end_idx > len(date_list):
                    continue

                idx_list = [(date_list[i], stock) for i in range(start_idx, end_idx)]
                if not all(i in dataset_index_set for i in idx_list):
                    continue

                result = dataset_index.get_indexer(idx_list)
                results.append(result)
            except Exception as e:
                print(f"Error in get_index for index {index}: {e}")

        if results:
            return np.stack(results)
        else:
            return None
    except Exception as e:
        print(f"Error in get_index_batch: {e}")
        return None

# 多线程处理
def multi_get_index(index_list, dataset, date_list, batch_size=10000):
    try:
        # 获取 CPU 核心数并设置线程数为 CPU 核心数减 1
        num_threads = max(1, os.cpu_count() - 1)
        print(f"Using {num_threads} threads for parallel processing.")

        # 预处理数据结构
        dataset_index_set, date_to_index, dataset_index = preprocess_data(dataset, date_list)

        # 初始化最终结果列表
        final_results = []

        # 将索引列表分成多个批次
        batches = [index_list[i:i + batch_size] for i in range(0, len(index_list), batch_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            print(f"Submitting tasks for {len(batches)} batches...")
            # 提交任务并获取未来对象
            futures = {executor.submit(get_index_batch, batch, dataset_index_set, date_to_index, dataset_index, date_list): batch for batch in tqdm(batches, desc="Submitting batches")}

            print("Processing batches...")
            # 获取结果并显示进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    batch_results = future.result()
                    if batch_results is not None and len(batch_results) > 0:
                        final_results.append(batch_results)
                except Exception as e:
                    print(f"Error in future for batch {futures[future]}: {e}")

        # 将所有批次的结果堆叠成一个数组
        if final_results:
            sorted_results = np.sort(final_results)
            print(f'最终输出结果大小为: {sorted_results.shape}')
            return sorted_results
        else:
            return np.array([])

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


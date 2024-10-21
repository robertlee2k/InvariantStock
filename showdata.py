import pandas as pd
import numpy as np

# 打印并保存 npy 文件的前100条内容到 CSV 文件
def print_and_save_npy_to_csv(data, csv_file, dataset):
    try:
        print(f"data shape is : {data.shape}")

        # 获取前100条数据
        top_100 = data[:100]
        print(f" top 100 raw data : {top_100}")

        # 将索引转换为日期和股票代码
        top_100_indices = [[(idx, dataset.index[idx][0], dataset.index[idx][1]) for idx in row] for row in top_100]

        # 创建 DataFrame
        df = pd.DataFrame(top_100_indices, columns=[f"Index_{i}" for i in range(top_100.shape[1])])

        # 打印前100条数据
        print(f"Top 100 rows of file  :")
        print(df.head(100))

        # 保存到 CSV 文件
        df.to_csv(csv_file, index=True)
        print(f"Saved top 100 rows to {csv_file}")

    except Exception as e:
        print(f"Error in print_and_save_npy_to_csv: {e}")


if __name__ == "__main__":
    dataset = pd.read_pickle(f"data/adataset-norm.pkl")
    # 打印包含 NaN 值的列名
    columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
    print("包含 NaN 值的列名:", columns_with_nan)

    # 打印包含 NaN 值的行
    rows_with_nan = dataset[dataset.isna().any(axis=1)]
    print("包含 NaN 值的行采样几条输出：")
    print(rows_with_nan)

    train_index = np.load(f"data/train_index.npy", allow_pickle=True)
    valid_index = np.load(f"data/valid_index.npy", allow_pickle=True)
    test_index = np.load(f"data/test_index.npy", allow_pickle=True)

    # 打印并保存前100条内容到 CSV 文件
    print_and_save_npy_to_csv(train_index, f'data/train_top_100.csv', dataset.index)
    print_and_save_npy_to_csv(valid_index, f'data/val_top_100.csv', dataset.index)
    print_and_save_npy_to_csv(test_index, f'data/test_top_100.csv', dataset.index)

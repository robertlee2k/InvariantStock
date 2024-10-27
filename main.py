import argparse
import multiprocessing
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import StockDataset, DynamicBatchSampler
from train_model import train_epoches, create_inv_predictor, create_env_predictor, create_feature_selection
from utils import set_seed, DataArgument, generate_prediction_scores,ModelManager
import wandb

def rankic(dataframe):
    # 打印原始数据框的前几行
    print("原始数据框的前几行:")
    print(dataframe.head())

    # 检查并处理缺失值
    if dataframe[['label', 'pred']].isnull().any().any():
        print("检测到缺失值，正在处理...")
        dataframe = dataframe.dropna(subset=['label', 'pred'])
        print("缺失值处理完成，数据框的前几行:")
        print(dataframe.head())
    else:
        print("未检测到缺失值")

    # 确保数据类型为数值
    dataframe['label'] = dataframe['label'].astype(float)
    dataframe['pred'] = dataframe['pred'].astype(float)
    print("数据类型转换完成，数据框的前几行:")
    print(dataframe.head())

    # 确保每个日期至少有两行数据
    date_counts = dataframe.index.get_level_values('date').value_counts()
    if (date_counts < 2).any():
        valid_dates = date_counts[date_counts >= 2].index
        dataframe = dataframe[dataframe.index.get_level_values('date').isin(valid_dates)]
        print("确保每个日期至少有两行数据，少于两行的日期drop掉了")
    else:
        print("所有日期都有至少两行数据，无需过滤")

    # 检查数据是否恒定
    def check_constant(df):
        if df['label'].nunique() == 1 or df['pred'].nunique() == 1:
            return False
        return True

    filtered_df = dataframe.groupby(level='date').filter(check_constant)
    if len(dataframe) != len(filtered_df):
        dataframe = filtered_df
        print("发现有日期的数据恒定，已做过滤，数据框的前几行:")
        print(dataframe.head())
    else:
        print("所有日期的数据都不恒定，无需过滤")

    # 计算 IC 和 Rank IC
    ic = dataframe.groupby(level='date').apply(lambda df: df["label"].corr(df["pred"]))
    ric = dataframe.groupby(level='date').apply(lambda df: df["label"].corr(df["pred"], method="spearman"))

    # 输出结果
    print("计算结果:")
    print({
        "IC": ic.mean(),
        "ICIR": ic.mean() / ic.std(),
        "Rank IC": ric.mean(),
        "Rank ICIR": ric.mean() / ric.std(),
    })

    return ric


def add_env(date, df):
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", ]
    begin_year = 2013
    end_year = 2024
    year = [str(i) for i in range(begin_year, end_year + 1)]
    df[month] = 0
    df[year] = 0
    df.loc[df.index.get_level_values("date") == date, month[date.month - 1]] = 1
    df.loc[df.index.get_level_values("date") == date, year[date.year - begin_year]] = 1
    return df


def multi_add_env(dataset):
    pool = multiprocessing.Pool()
    results = pool.starmap(add_env, [*dataset.groupby("date")])
    pool.close()
    pool.join()
    results = [i for i in results if i is not None]
    return pd.concat(results)


def main():
    parser = argparse.ArgumentParser(description='Train a predictor model on stock data')
    parser.add_argument('--train', type=bool, default=False, help='do training or do predicting')
    parser.add_argument('--num_epochs', type=int, default=12, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=300, help='batch size')
    parser.add_argument('--feat_dim', type=int, default=20, help='features dimension')
    parser.add_argument('--seq_len', type=int, default=60, help='sequence length')
    parser.add_argument('--factor_dim', type=int, default=10, help='number of factors')
    parser.add_argument('--hidden_dim', type=int, default=20, help='hidden variables dimension')
    parser.add_argument('--seed', type=int, default=88, help='random seed')
    parser.add_argument('--run_name', type=str, default="Invariant", help='name of the run')
    parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
    parser.add_argument('--wandb', action='store_false', default=True, help='whether to use wandb')
    parser.add_argument('--normalize', action='store_true', help='whether to normalize the data')
    parser.add_argument('--device', default="cuda:0", type=str, help='devices')

    #TODO data_args needed to be removed
    data_args = DataArgument(use_qlib=False, normalize=True, select_feature=False)

    dataset = pd.read_pickle(f"{data_args.save_dir}/adataset-norm.pkl")
    # 删除不想要的列
    delete_column = ['adjustflag']
    # 检查待删除的列是否存在于数据集中
    existing_columns = [col for col in delete_column if col in dataset.columns]
    if existing_columns:
        dataset = dataset.drop(existing_columns, axis=1)
    # 打印包含 NaN 值的列名
    columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
    if len(columns_with_nan) > 0:
        print("包含 NaN 值的列名:", columns_with_nan)
        # 删除包含 NaN 值的列
        dataset = dataset.drop(columns=columns_with_nan)
        # 打印删除后的数据集的前几行
        print("删除包含 NaN 值 和不想要的列后的数据集，再看看有没有NaN值:")
        # 打印包含 NaN 值的行
        rows_with_nan = dataset[dataset.isna().any(axis=1)]
        print("包含 NaN 值的行采样几条输出：")
        print(rows_with_nan)
    else:
        print("已校验，数据集里没有NaN值,数据集描述如下")
        print(dataset.describe())
    print(f"数据集列名：{dataset.columns}")
    train_index = np.load(f"{data_args.save_dir}/train_index.npy", allow_pickle=True)
    valid_index = np.load(f"{data_args.save_dir}/valid_index.npy", allow_pickle=True)
    test_index = np.load(f"{data_args.save_dir}/test_index.npy", allow_pickle=True)

    args = parser.parse_args()
    args.save_dir = args.save_dir + "/" + str(args.factor_dim)
    args.feat_dim = len(dataset.columns) - 1
    if args.wandb:
        print("初始化 wandb")
        wandb.init(project="InvariantStock", config=args, name=f"{args.run_name}")
        wandb.config.update(args)
    dataset = multi_add_env(dataset)
    dataset = dataset.astype("float")
    args.env_size = len(dataset.columns) - args.feat_dim - 1
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # create dataloaders
    # Dynamic batch size
    train_ds = StockDataset(dataset, train_index)
    valid_ds = StockDataset(dataset, valid_index)
    test_ds = StockDataset(dataset, valid_index)
    train_batch_sizes = pd.DataFrame([i[0] for i in dataset.index[train_index[:, 0]].values]).value_counts(
        sort=False).values
    train_batch_sampler = DynamicBatchSampler(train_ds, train_batch_sizes)
    valid_batch_sizes = pd.DataFrame([i[0] for i in dataset.index[valid_index[:, 0]].values]).value_counts(
        sort=False).values
    valid_batch_sampler = DynamicBatchSampler(valid_ds, valid_batch_sizes)
    test_batch_sizes = pd.DataFrame([i[0] for i in dataset.index[test_index[:, 0]].values]).value_counts(
        sort=False).values
    test_batch_sampler = DynamicBatchSampler(test_ds, test_batch_sizes)
    print(f"训练数据批次{len(train_batch_sizes)}, train_batch_sizes={train_batch_sizes} ")
    print(f"验证数据批次{len(valid_batch_sizes)}, valid_batch_sizes={valid_batch_sizes} ")
    print(f"测试数据批次{len(test_batch_sizes)}, test_batch_sizes={test_batch_sizes}")
    train_dataloader = DataLoader(train_ds, batch_sampler=train_batch_sampler, shuffle=False, num_workers=4)
    valid_dataloader = DataLoader(valid_ds, batch_sampler=valid_batch_sampler, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_ds, batch_sampler=test_batch_sampler, shuffle=False, num_workers=4)
    # create model
    feature_mask, feature_reconstructor = create_feature_selection(args)
    # environment-agnostic module
    predictor = create_inv_predictor(args)
    # environment-awareness module
    env_predictor = create_env_predictor(args)
    device = args.device
    feature_mask.to(device)
    feature_reconstructor.to(device)
    predictor.to(device)
    env_predictor.to(device)
    model_manager = ModelManager(save_dir=args.save_dir)
    if args.train:
        train_epoches(args, model_manager, env_predictor, feature_mask, feature_reconstructor, predictor,
                      train_dataloader, valid_dataloader)
    # 获取最佳模型的字典
    feat_mask_dict, predictor_dict = model_manager.get_best_model_dicts()
    # loading the best model for the final test set
    predictor.load_state_dict(predictor_dict)
    feature_mask.load_state_dict(feat_mask_dict)
    output = generate_prediction_scores(feature_mask, predictor, test_dataloader, args)
    # output 是一个带有复合索引 (date, code) 的 DataFrame
    output.index = dataset.index[test_index[:, -1]]
    output["label"] = dataset.loc[output.index, 'label']
    print("Test IC Result:")
    rankic(output)
    # 按 date 分组，并在每个分组内根据 pred 值从大到小排序
    output['pred_rank'] = output.groupby(level='date')['pred'].rank(ascending=False, method='first')
    # 重置索引，将复合索引转换为普通列
    output_reset = output.reset_index()
    # 保存到 CSV 文件
    output_reset.to_csv('predict_results.csv', index=False)
    print("保存预测结果文件成功.")
    if args.wandb:
        wandb.log({"Best Validation RankIC": best_rankic})
        wandb.finish()


if __name__ == '__main__':
    main()

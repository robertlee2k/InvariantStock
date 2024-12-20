import pandas as pd
import numpy as np
import random
import os
from dataclasses import dataclass, field
from Layers import *
from tqdm.auto import tqdm
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DataArgument:
    save_dir: str = field(
        default='./data',
        metadata={"help": 'directory to save model'}
    )
    start_time: str = field(
        default="2013-01-01",
        metadata={"help": "start_time"}
    )
    end_time: str = field(
        default='2024-12-31',
        metadata={"help": "end_time"}
    )

    fit_end_time: str = field(
        default="2022-12-31",
        metadata={"help": "fit_end_time"}
    )

    val_start_time: str = field(
        default='2023-01-01',
        metadata={"help": "val_start_time"}
    )

    val_end_time: str = field(default='2023-12-31')

    seq_len: int = field(default=20)

    normalize: bool = field(
        default=True,
    )
    select_feature: bool = field(
        default=True,
    )
    use_qlib: bool = field(
        default=False,
    )


def load_model(args):
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent,
                                   hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_prior_model = FatorPrior(args.batch_size, args.hidden_size, args.num_factor)
    predictor = Predictor(feature_extractor, factor_encoder, factor_decoder, factor_prior_model, args)

    return predictor


@torch.no_grad()
def generate_prediction_scores(masker, model, test_dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理预测使用设备： {device}")
    model.to(device)
    masker.to(device)
    model.eval()
    masker.eval()
    ls = []
    with tqdm(total=len(test_dataloader)) as pbar:
        for i, (char, _) in (enumerate(test_dataloader)):
            char = char.float().to(device)
            if char.shape[1] != args.seq_len:
                print(f"预测数据形状{char.shape[1]}与 设定的序列长度{args.seq_len}不匹配，跳过该批次")
                continue
            char = char[..., :args.feat_dim]
            mask = masker(char.float())[..., 0]
            feature = mask * char
            predictions = model.prediction(feature)
            df = pd.DataFrame(predictions.cpu().numpy(), columns=['pred'])
            pbar.update(1)
            ls.append(df)

    return pd.concat(ls, ignore_index=True)


@dataclass
class test_args:
    run_name: str
    num_factor: int
    normalize: bool = True
    select_feature: bool = True

    batch_size: int = 300
    seq_length: int = 20

    hidden_size: int = 20
    num_latent: int = 24

    save_dir = './best_model'
    use_qlib: bool = False
    device = "cuda:0"


class ModelManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_rankic = -float('inf')
        self.csv_path = os.path.join(save_dir, 'best_model.csv')

    def save_model_if_better(self, predictor, feature_mask, run_name, epoch, rankic):
        if rankic > self.best_rankic:
            self.best_rankic = rankic
            predictor_root = os.path.join(self.save_dir, f'best_predictor_{run_name}_{epoch}.pt')
            feat_mask_root = os.path.join(self.save_dir, f'best_feat_mask_{run_name}_{epoch}.pt')

            torch.save(predictor.state_dict(), predictor_root)
            torch.save(feature_mask.state_dict(), feat_mask_root)

            # 更新 best_model.csv
            model_info = {
                'predictor_root': [predictor_root],
                'feat_mask_root': [feat_mask_root],
                'rankic': [rankic]
            }
            df = pd.DataFrame(model_info)
            df.to_csv(self.csv_path, index=False, header=True)

    def get_best_model_dicts(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} does not exist.")

        df = pd.read_csv(self.csv_path, encoding='utf8')
        predictor_root = df['predictor_root'].iloc[0]
        feat_mask_root = df['feat_mask_root'].iloc[0]

        predictor_dict = torch.load(predictor_root, weights_only=True)
        feat_mask_dict = torch.load(feat_mask_root, weights_only=True)

        return feat_mask_dict, predictor_dict

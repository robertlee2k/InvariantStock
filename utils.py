import torch
import numpy as np
import random
import os
from dataclasses import dataclass, field
from Layers import *
from tqdm.auto import tqdm

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
        default="2010-12-01",
        metadata={"help": "start_time"}
    )
    end_time: str =field(
        default='2020-12-31', 
        metadata={"help": "end_time"}
    )

    fit_end_time: str= field(
        default="2017-12-31", 
        metadata={"help": "fit_end_time"}
    )

    val_start_time : str = field(
        default='2018-01-01', 
        metadata={"help": "val_start_time"}
    )

    val_end_time: str =field(default='2018-12-31')

    seq_len : int = field(default=20)

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
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.batch_size, args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor,args)
    
    return factorVAE


def single_pred(model, test_dataloader, test_dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for i, (char, _) in (enumerate(test_dataloader)):
        char = char.to(device)
        if char.shape[1] != args.seq_length:
            return
        
        predictions = model.prediction(char.float())
        return predictions
    
    return
            
@torch.no_grad()
def generate_prediction_scores(masker,model, test_dataloader, test_dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # extractor.to(device)
    masker.to(device)
    
    model.eval()
    # extractor.eval()
    masker.eval()
    test_loss = 0
    ls = []
    mask_ls = []
    with tqdm(total=len(test_dataloader)) as pbar:
        for i, (char, _) in (enumerate(test_dataloader)):
            char = char.float().to(device)
            if char.shape[1] != args.seq_length:
                continue
            mask = masker(char.float())[...,0]
            feature = mask * char
            predictions = model.prediction(feature)
            df = pd.DataFrame(predictions.cpu().numpy(), columns=['score'])
            # mask = pd.DataFrame(mask.squeeze().cpu().numpy())
            # try:
            #     index = test_dataset.index[(args.seq_length +i -1) * args.batch_size : (args.seq_length+i) * args.batch_size]
            #     df.index = index
            #     df.drop('empty', level='instrument',inplace=True)
            #     ls.append(df)
            # except:
            #     err.append(df)
            pbar.update(1)
            ls.append(df)
            # mask_ls.append(mask)
    return pd.concat(ls,ignore_index=True)
# ,pd.concat(mask_ls,ignore_index=True)

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
    
    save_dir='./best_model'
    use_qlib: bool = False
    device="cuda:0"
    

    
    
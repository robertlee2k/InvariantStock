import pandas as pd
import torch
import torch.nn.functional as F
import os
from tqdm.auto import tqdm

from Layers import FeatureReconstructor, Predictor, FeatureMask, FeatureExtractor, FactorDecoder, FactorEncoder, \
    FatorPrior, AlphaLayer, BetaLayer


class NaNException(Exception):
    """自定义异常类，用于处理NaN值"""
    pass


def check_nan(tensor, name):
    """
    检查张量中是否存在NaN值，并记录相关信息。

    :param tensor: 需要检查的张量
    :param name: 张量的名称，用于记录信息
    """
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            print(f"警告：{name} 包含 NaN 值")
            # 打印整个张量
            print(f"张量 {name} 的所有行：\n{tensor}")
            # 将包含 NaN 值的行输出到 na-values.csv
            tensor_df = pd.DataFrame(tensor.detach().cpu().numpy().reshape(tensor.size(0), -1))
            tensor_df.to_csv('na-all-values.csv', index=False)
            raise NaNException(f"{name} 包含 NaN 值")


def print_tensor_stats(tensor, batch_num):
    print(f"Batch {batch_num} 输入数据分布情况 - mean: {tensor.mean().item()}, std: {tensor.std().item()}, "
          f"min: {tensor.min().item()}, max: {tensor.max().item()}")


def train(feature_reconstructor, feature_mask, factorVAE, env_factorVAE, train_dataloader, featrue_optimizer, optimizer,
          env_optimizer, featrue_scheduler, scheduler, env_scheduler, args, epoch=0):
    device = args.device
    feature_mask.to(device)
    feature_reconstructor.to(device)
    feature_reconstructor.train()
    factorVAE.to(device)
    factorVAE.train()
    env_factorVAE.to(device)
    env_factorVAE.train()
    total_loss = 0
    total_env_loss = 0
    total_diff_loss = 0
    total_pred_loss = 0
    total_env_pred_loss = 0
    total_self_pred_loss = 0
    total_recon_diff_loss = 0
    total_kl_diff_loss = 0
    total_rank_loss = 0
    total_env_rank_loss = 0
    total_kl_loss = 0
    total_env_kl_loss = 0
    total_rank_diff_loss = 0
    path = epoch % 3
    batch_count = 0  # 初始化批次计数器
    with tqdm(total=len(train_dataloader), desc=f"正在训练") as pbar:
        for char, returns in train_dataloader:
            batch_count += 1  # 每次迭代时递增计数器
            if char.shape[1] != args.seq_len:
                print(f"训练数据形状{char.shape[1]}与 设定的序列长度{args.seq_len}不匹配，跳过该批次")
                continue
            inputs = char.to(device)
            labels = returns[:, -1].reshape(-1, 1).to(device)
            inputs = inputs.float()
            labels = labels.float()

            mask = feature_mask(inputs[..., :-args.env_size])[..., 0]
            new_features = mask * inputs[..., :-args.env_size]
            self_recondstruction = feature_reconstructor(new_features)

            if torch.isnan(torch.sum(new_features)):
                print(epoch)

            env = inputs[:, :, -args.env_size:]
            env_new_feture = torch.cat([new_features, env], dim=-1)
            batch_size = inputs.shape[0]

            loss, pred_loss, rank_loss, kl_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factorVAE(
                new_features, labels)
            check_nan(loss, "loss")

            # # 打印输入数据的描述情况
            # print_tensor_stats(env_new_feture,batch_count)
            try:
                env_loss, env_pred_loss, env_rank_loss, env_kl_loss, env_reconstruction, env_factor_mu, env_factor_sigma, env_pred_mu, env_pred_sigma = env_factorVAE(
                    env_new_feture, labels)
                check_nan(env_loss, "env_loss")

            except NaNException as nanE:
                print("打印变换后的输入矩阵:")
                env_new_feture_df = pd.DataFrame(
                    env_new_feture.detach().cpu().numpy().reshape(env_new_feture.size(0), -1))
                print(env_new_feture_df)
                env_new_feture_df.to_csv('env_new_feture.csv', index=False)

                raise Exception(f"遇到 NaN 值，训练爆了")

            if path == 0:
                # feature selection
                ranks = torch.ones_like(labels)
                ranks_index = torch.argsort(labels, dim=0)
                ranks[ranks_index, 0] = torch.arange(0, batch_size).reshape(ranks.shape).float().to(device)
                ranks = (ranks - torch.mean(ranks)) / torch.std(ranks)
                ranks = ranks ** 2
                all_ones = torch.ones(batch_size, 1).to(device)
                pre_dif = (torch.matmul(reconstruction, torch.transpose(all_ones, 0, 1))
                           - torch.matmul(all_ones, torch.transpose(reconstruction, 0, 1)))
                env_pre_dif = (
                        torch.matmul(all_ones, torch.transpose(env_reconstruction, 0, 1)) -
                        torch.matmul(env_reconstruction, torch.transpose(all_ones, 0, 1))
                )
                rank_diff_loss = torch.mean(ranks * F.relu(pre_dif * env_pre_dif))

                featrue_optimizer.zero_grad()
                self_pred_loss = F.mse_loss(inputs[..., :-args.env_size], self_recondstruction)
                recon_diff_loss = F.mse_loss(reconstruction, env_reconstruction)
                kl_diff_loss = KL_Divergence(pred_mu, pred_sigma, env_pred_mu, env_pred_sigma)
                diff_loss = self_pred_loss + recon_diff_loss + kl_diff_loss + rank_diff_loss

                total_diff_loss += diff_loss.item() * inputs.size(0)
                total_rank_diff_loss += rank_diff_loss.item() * inputs.size(0)
                total_self_pred_loss += self_pred_loss.item() * inputs.size(0)
                total_recon_diff_loss += recon_diff_loss.item() * inputs.size(0)
                total_kl_diff_loss += kl_diff_loss.item() * inputs.size(0)

                check_nan(diff_loss, "diff_loss")

                print(f"第{epoch} 轮的第{batch_count} 个batch的loss为:{diff_loss.item()}")
                diff_loss.backward()
                featrue_optimizer.step()
                featrue_scheduler.step()
            elif path in [1]:
                # without env
                optimizer.zero_grad()

                total_loss += loss.item() * inputs.size(0)
                total_pred_loss += pred_loss.item() * inputs.size(0)
                total_kl_loss += kl_loss.item() * inputs.size(0)
                total_rank_loss += rank_loss.item() * inputs.size(0)

                # 裁剪梯度，max_norm 可以根据需要调整
                print(f"第{epoch} 轮的第{batch_count} 个batch的loss为:{loss.item()}")
                torch.nn.utils.clip_grad_norm_(factorVAE.parameters(), max_norm=0.5)
                loss.backward()
                optimizer.step()
                scheduler.step()

            elif path in [2]:
                # with env
                env_optimizer.zero_grad()
                total_env_loss += env_loss.item() * inputs.size(0)
                total_env_pred_loss += env_pred_loss.item() * inputs.size(0)
                total_env_rank_loss += env_rank_loss.item() * inputs.size(0)
                total_env_kl_loss += env_kl_loss.item() * inputs.size(0)

                print(f"第{epoch} 轮的第{batch_count} 个batch的环境感知loss为:{env_loss.item()}")
                print(f"      其中：重建损失为：{env_pred_loss}, 排序损失为：{env_rank_loss}, KL损失为：{env_kl_loss}")
                env_loss.backward()

                # 裁剪梯度，max_norm 可以根据需要调整
                torch.nn.utils.clip_grad_norm_(env_factorVAE.parameters(), max_norm=0.1)

                # 检查每个参数的梯度是否为 NaN
                for name, param in env_factorVAE.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Gradient for {name} is NaN")
                env_optimizer.step()
                env_scheduler.step()

            pbar.update(1)
        # print(loss)
    avg_loss = total_loss / len(train_dataloader.dataset)
    env_avg_loss = total_env_loss / len(train_dataloader.dataset)
    diff_avg_loss = total_diff_loss / len(train_dataloader.dataset)
    avg_pred_loss = total_pred_loss / len(train_dataloader.dataset)
    avg_env_pred_loss = total_env_pred_loss / len(train_dataloader.dataset)
    avg_self_pred_loss = total_self_pred_loss / len(train_dataloader.dataset)
    avg_recon_diff_loss = total_recon_diff_loss / len(train_dataloader.dataset)
    avg_kl_diff_loss = total_kl_diff_loss / len(train_dataloader.dataset)
    avg_rank_loss = total_rank_loss / len(train_dataloader.dataset)
    avg_env_rank_loss = total_env_rank_loss / len(train_dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(train_dataloader.dataset)
    avg_env_kl_loss = total_env_kl_loss / len(train_dataloader.dataset)
    avg_rank_diff_loss = total_rank_diff_loss / len(train_dataloader.dataset)
    return avg_loss, avg_pred_loss, env_avg_loss, avg_env_pred_loss, diff_avg_loss, avg_self_pred_loss, avg_recon_diff_loss, avg_kl_diff_loss, avg_rank_loss, avg_env_rank_loss, avg_kl_loss, avg_env_kl_loss, avg_rank_diff_loss


@torch.no_grad()
def validate(feature_mask, factorVAE, dataloader, args):
    device = args.device
    factorVAE.to(device)
    factorVAE.eval()
    total_loss = 0
    pred_loss = 0
    total_rank_loss = 0
    total_env_kl_loss = 0
    total_rankic = 0
    with tqdm(total=len(dataloader), desc=f"验证阶段") as pbar:
        for char, returns in dataloader:
            if char.shape[1] != args.seq_len:
                continue
            inputs = char.to(device)
            labels = returns[:, -1].reshape(-1, 1).to(device)
            inputs = inputs.float()
            labels = labels.float()

            if torch.isnan(inputs).any():
                print("注意：验证数据，Inputs contain NaN values")
            if torch.isnan(labels).any():
                print("注意：验证数据，Labels contain NaN values")

            batch_size = inputs.shape[0]
            mask = feature_mask(inputs[..., :-args.env_size])[..., 0]
            new_features = mask * inputs[..., :-args.env_size]

            if torch.isnan(new_features).any():
                print("注意：验证数据，New features contain NaN values")

            loss, reconstruction_loss, rank_loss, env_kl_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factorVAE(
                new_features, labels)
            pred = factorVAE.prediction(new_features)

            if torch.isnan(pred).any():
                print("注意：验证阶段 Prediction contains NaN values")

            total_loss += loss.item() * inputs.size(0)
            pred_loss += reconstruction_loss.item() * inputs.size(0)
            total_rank_loss += rank_loss.item() * inputs.size(0)
            total_env_kl_loss += env_kl_loss.item() * inputs.size(0)

            ranks = pred_ranks = torch.ones_like(labels)
            ranks_index = torch.argsort(labels, dim=0)
            ranks[ranks_index, 0] = torch.arange(0, batch_size).reshape(ranks.shape).float().to(device)
            ranks = (ranks - torch.mean(ranks)) / torch.std(ranks)

            pred_ranks_index = torch.argsort(pred, dim=0)
            pred_ranks[pred_ranks_index, 0] = torch.arange(0, batch_size).reshape(ranks.shape).float().to(device)
            pred_ranks = (pred_ranks - torch.mean(pred_ranks)) / torch.std(pred_ranks)

            rankic = (ranks * pred_ranks).mean()
            total_rankic += rankic.item() * inputs.size(0)

            pbar.update(1)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_pred_loss = pred_loss / len(dataloader.dataset)
    avg_rank_loss = total_rank_loss / len(dataloader.dataset)
    avg_env_kl_loss = total_env_kl_loss / len(dataloader.dataset)
    avg_rankic = total_rankic / len(dataloader.dataset)

    return avg_loss, avg_pred_loss, avg_rank_loss, avg_env_kl_loss, avg_rankic


def KL_Divergence(mu1, sigma1, mu2, sigma2):
    kl_div = (torch.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5).sum()
    return kl_div


def create_env_predictor(args):
    env_feature_extractor = FeatureExtractor(feat_dim=args.feat_dim + args.env_size, hidden_dim=args.hidden_dim)
    env_factor_encoder = FactorEncoder(factor_dims=args.factor_dim, num_portfolio=args.feat_dim,
                                       hidden_dim=args.hidden_dim)
    env_alpha_layer = AlphaLayer(args.hidden_dim)
    env_beta_layer = BetaLayer(args.hidden_dim, args.factor_dim)
    env_factor_decoder = FactorDecoder(env_alpha_layer, env_beta_layer)
    env_factor_prior_model = FatorPrior(args.batch_size, args.hidden_dim, args.factor_dim)
    env_predictor = Predictor(env_feature_extractor, env_factor_encoder, env_factor_decoder, env_factor_prior_model,
                              args)
    return env_predictor


def create_inv_predictor(args):
    feature_extractor = FeatureExtractor(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim)
    factor_encoder = FactorEncoder(factor_dims=args.factor_dim, num_portfolio=args.feat_dim, hidden_dim=args.hidden_dim)
    alpha_layer = AlphaLayer(args.hidden_dim)
    beta_layer = BetaLayer(args.hidden_dim, args.factor_dim)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_prior_model = FatorPrior(args.batch_size, args.hidden_dim, args.factor_dim)
    predictor = Predictor(feature_extractor, factor_encoder, factor_decoder, factor_prior_model, args)
    return predictor


def create_feature_selection(args):
    # feature selection module
    feature_reconstructor = FeatureReconstructor(feat_dim=args.feat_dim)
    feature_mask = FeatureMask(feat_dim=args.feat_dim, hidden_dim=args.feat_dim)
    return feature_mask, feature_reconstructor


def train_epoches(args, model_manager, env_predictor, feature_mask, feature_reconstructor, predictor,
                  train_dataloader, valid_dataloader):
    ## 构建训练优化器
    featrue_optimizer = torch.optim.Adam(list(feature_reconstructor.parameters()) + list(feature_mask.parameters()),
                                         lr=args.lr)
    featrue_scheduler = torch.optim.lr_scheduler.OneCycleLR(featrue_optimizer, pct_start=0.1, max_lr=args.lr,
                                                            steps_per_epoch=len(train_dataloader),
                                                            epochs=args.num_epochs // 3)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                    steps_per_epoch=len(train_dataloader), epochs=args.num_epochs // 3)

    env_optimizer = torch.optim.Adam(env_predictor.parameters(), lr=args.lr)
    env_scheduler = torch.optim.lr_scheduler.OneCycleLR(env_optimizer, max_lr=args.lr, pct_start=0.1,
                                                        steps_per_epoch=len(train_dataloader),
                                                        epochs=args.num_epochs // 3)

    # Start Training
    for epoch in tqdm(range(args.num_epochs), desc=f"整体训练"):
        (train_loss, pred_loss, env_loss, env_pred_loss, diff_loss, self_pred_loss, recon_diff_loss,
         kl_diff_loss, rank_loss, env_rank_loss, kl_loss, env_kl_loss, rank_diff_loss) = train(
            feature_reconstructor, feature_mask, predictor, env_predictor, train_dataloader, featrue_optimizer,
            optimizer, env_optimizer, featrue_scheduler, scheduler, env_scheduler, args, epoch=epoch)
        val_loss, val_pred_loss, val_rank_loss, val_kl_loss, avg_rankic = validate(feature_mask, predictor,
                                                                                   valid_dataloader, args)
        path = epoch % 3
        print(f"Epoch 第{epoch + 1}轮: 验证数据集上损失 ",
              {"Validation Toal Loss": round(val_loss, 6), "Validation Pred Loss": round(val_pred_loss, 6),
               "Validation Ranking Loss": round(val_rank_loss, 6), "Validation KL Loss": round(val_kl_loss, 6),
               "Validation RankIC": round(avg_rankic, 6)})
        # if args.wandb:
        #     wandb.log({"Validation Toal Loss": val_loss, "Validation Pred Loss": val_pred_loss,
        #                "Validation Ranking Loss": val_rank_loss, "Validation KL Loss": val_kl_loss,
        #                "Validation RankIC": avg_rankic}, step=epoch)
        #     if path == 0:
        #         wandb.log({"Different Loss": diff_loss, "Self Reconstruction Loss": self_pred_loss,
        #                    "Reconstruction Diff Loss": recon_diff_loss, "KL Diff Loss": kl_diff_loss}, step=epoch)
        #     elif path in [1]:
        #         wandb.log({"No Env Loss": train_loss, "No Env Pred Loss": pred_loss, "No Env Ranking Loss": rank_loss,
        #                    "No Env KL Loss": kl_loss}, step=epoch)
        #     elif path in [2]:
        #         wandb.log({"With Env Loss": env_loss, "With Env Pred Loss": env_pred_loss,
        #                    "With Env Ranking Loss": env_rank_loss, "With Env KL Loss": env_kl_loss}, step=epoch)

        if path == 0:
            print(f"Epoch 第{epoch + 1}轮: 参数选择训练损失",
                  {"Different Loss": round(diff_loss, 6), "Self Reconstruction Loss": round(self_pred_loss, 6),
                   "Reconstruction Diff Loss": round(recon_diff_loss, 6), "KL Diff Loss": round(kl_diff_loss, 6),
                   "Rank Diff Loss": round(rank_diff_loss, 6)})
        elif path in [1]:
            print(f"Epoch {epoch + 1}: 环境无关训练损失",
                  {"No Env Loss": round(train_loss, 6), "No Env Pred Loss": round(pred_loss, 6),
                   "No Env Ranking Loss": round(rank_loss, 6), "No Env KL Loss": round(kl_loss, 6)})
        elif path in [2]:
            print(f"Epoch 第{epoch + 1}轮: 环境感知训练损失",
                  {"With Env Loss": round(env_loss, 6), "With Env Pred Loss": round(env_pred_loss, 6),
                   "With Env Ranking Loss": round(env_rank_loss, 6), "With Env KL Loss": round(env_kl_loss, 6)})

        model_manager.save_model_if_better(predictor=predictor, feature_mask=feature_mask, epoch=epoch,
                                           run_name=args.run_name, rankic=avg_rankic)


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

        df = pd.read_csv(self.csv_path)
        predictor_root = df['predictor_root'].iloc[0]
        feat_mask_root = df['feat_mask_root'].iloc[0]

        predictor_dict = torch.load(predictor_root, weights_only=True)
        feat_mask_dict = torch.load(feat_mask_root, weights_only=True)

        return feat_mask_dict, predictor_dict

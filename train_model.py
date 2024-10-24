import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import Layers
from Layers import NaNException
import pandas as pd

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
    with tqdm(total=len(train_dataloader)) as pbar:
        for char, returns in train_dataloader:
            batch_count += 1  # 每次迭代时递增计数器
            if char.shape[1] != args.seq_len:
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
            Layers.check_nan(loss,"loss")

            # # 打印输入数据的描述情况
            # print_tensor_stats(env_new_feture,batch_count)
            try:
                env_loss, env_pred_loss, env_rank_loss, env_kl_loss, env_reconstruction, env_factor_mu, env_factor_sigma, env_pred_mu, env_pred_sigma = env_factorVAE(
                    env_new_feture, labels)
                Layers.check_nan(env_loss, "env_loss")

            except NaNException as nanE:
                print("打印变换后的输入矩阵:")
                env_new_feture_df = pd.DataFrame(env_new_feture.detach().cpu().numpy().reshape(env_new_feture.size(0), -1))
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

                Layers.check_nan(diff_loss, "diff_loss")

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
    with tqdm(total=len(dataloader)) as pbar:
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

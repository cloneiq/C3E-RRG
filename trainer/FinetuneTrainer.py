"""
Finetune trainer for report generation
"""
import pandas as pd
from trainer.BaseTrainer import BaseTrainer
import numpy as np
import torch
import time
from trainer.PretrainTrainer import unpatchify, vis_heatmap
from utils.loss import patchify
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.loss import dymes_total_loss
from utils.vis_utils import export_epoch_visuals, export_x_token_attn_maps, export_grad_visuals
from torch.nn.utils import clip_grad_norm_

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.best_score = 0.

        # 预训练权重
        if args.get("load_model_path", None):
            pretrained_dict = torch.load(args["load_model_path"], map_location='cuda')['state_dict']
            self.model.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded pretrained model from {args['load_model_path']}")
        else:
            print(" No pretrained model provided. Training from scratch.")

        self._vis_every = int(self.args.get("vis_every", 1))  # 每多少个 epoch 导出一次
        self._vis_grad_every = int(self.args.get("vis_grad_every", 0))  # e.g., 5 表示每5个epoch导出一次梯度图
        
        # CHI 保存相关
        self._chi_save_every = int(args.get("chi_save_every", 200))
        self._chi_cmap = args.get("chi_cmap", "jet")
        self._chi_dir = args.get("chi_save_dir", "")
        # 若未设置，落到ckpt目录下
        if not self._chi_dir:
            self._chi_dir = os.path.join(self.checkpoint_dir, "chi_maps")
        os.makedirs(self._chi_dir, exist_ok=True)

    def _save_chi_heatmap(self, chi_map: torch.Tensor, epoch: int, batch_idx: int, prefix: str = "train"):
        """
        chi_map: Tensor[B,H,W] (CPU) 取第一个样本保存
        """
        if chi_map is None or not isinstance(chi_map, torch.Tensor):
            return
        try:
            arr = chi_map[0].numpy()
            plt.figure(figsize=(3.6, 3.6), dpi=120)
            plt.imshow(arr, cmap=self._chi_cmap)
            plt.axis('off')
            out_path = os.path.join(self._chi_dir, f"{prefix}_e{epoch:03d}_b{batch_idx:06d}.png")
            plt.tight_layout(pad=0.1)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            print(f"[WARN] save CHI heatmap failed: {e}")

    def _train_epoch(self, epoch):
        train_loss = 0
        # ===== DyMES: 全模块非零起步 + 分段增重（MIMIC 更稳） =====
        if hasattr(self.model, "lambda_cem"):
            if epoch <= 3:
                self.model.lambda_cem = 0.001
                self.model.lambda_back = 0.0005
                self.model.lambda_cf = 0.005
                self.model.lambda_fd_ortho = 0.0005
            elif epoch <= 15:
                self.model.lambda_cem = 0.002
                self.model.lambda_back = 0.002
                self.model.lambda_cf = 0.02
                self.model.lambda_fd_ortho = 0.002
            else:
                self.model.lambda_cem = 0.002
                self.model.lambda_back = 0.001
                self.model.lambda_cf = 0.001
                self.model.lambda_fd_ortho = 0.001

        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()

            # 1) 前向
            output = self.model(images, reports_ids, mode='train')

            # 2) 统一总损失（含 CE + CEM + back + CF + FD_ortho）
            loss, terms = dymes_total_loss(
                self.args, output, reports_ids, reports_masks,
                model=self.model, criterion=self.criterion
            )
            # 3) 训练中按频率保存 CHI 热力图（沿用原逻辑）
            if isinstance(output, list) and len(output) > 2 and isinstance(output[2], dict):
                chi_map = output[2].get('chi_map', None)
                global_step = output[2].get('global_step', None)
                if chi_map is not None and isinstance(global_step, int):
                    if (global_step % self._chi_save_every) == 0:
                        self._save_chi_heatmap(chi_map, epoch, batch_idx, prefix="train")

            # 4) 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            print(f"\repoch: {epoch} {batch_idx}/{len(self.train_dataloader)}\t"
                  f"loss: {loss:.3f}\tmean loss: {train_loss/(batch_idx+1):.3f}",
                  flush=True, end='')

            # 5) 学习率调度
            if self.args["lr_scheduler"] != 'StepLR':
                self.lr_scheduler.step()
        if self.args["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("\n")
        print("\tEpoch {}\tmean_loss: {:.4f}\ttime: {:.4f}s".format(epoch, log['train_loss'], time.time() - start_time))

        # ===== 验证 =====
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], [] # 存储真实报告和生成报告
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            for k, v in val_met.items():
                self.monitor.logkv(key='val_' + k, val=v)
            val_met['p'] = lp
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        # ===== 测试 =====
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            test_subject_ids, test_study_ids = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                # images_id: batch list，元素可能是 (subject_id, study_id)
                for x in images_id:
                    if isinstance(x, (list, tuple)) and len(x) == 2:
                        test_subject_ids.append(int(x[0]))
                        test_study_ids.append(int(x[1]))
                    else:
                        test_subject_ids.append(-1)
                        test_study_ids.append(-1)

                p = torch.cat([p, output])
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            for k, v in test_met.items():
                self.monitor.logkv(key='test_' + k, val=v)
            test_met['p'] = lp
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        # ===== 只在“最佳”或 inference 时写 CSV（覆盖写：只保留一行） =====
        try:
            monitor_metric = self.args.get("monitor_metric", "BLEU_4")
            monitor_mode   = self.args.get("monitor_mode", "max")   # 'max' or 'min'
            task_kind      = self.args.get("task", "finetune")
            # 当前 epoch 的验证集监控值
            cur_val = self.monitor.name2val.get(f"val_{monitor_metric}", None)
            # 初始化 best_score
            if not hasattr(self, "_best_score"):
                self._best_score = -float("inf") if monitor_mode == "max" else float("inf")
            # 是否需要写：inference 任务恒写；finetune 仅在更优时写
            force_write = (task_kind == "inference")
            is_better   = False
            if (cur_val is not None) and not force_write:
                is_better = (cur_val > self._best_score) if monitor_mode == "max" else (cur_val < self._best_score)
            if force_write or is_better:
                if not force_write:
                    self._best_score = cur_val
                # 1) 写测试集预测文本（覆盖写）
                # pred_csv = os.path.join(self.args["result_dir"], "predictions_test.csv")
                # pd.DataFrame({"Report Impression": test_res}).to_csv(pred_csv, index=False)
                pred_csv = os.path.join(self.args["result_dir"], "predictions_test_with_id.csv")
                pd.DataFrame({
                    "subject_id": test_subject_ids,
                    "study_id": test_study_ids,
                    "Report Impression": test_res,
                    # 可选：把 GT 一起存，方便排查/CE
                    "GT": test_gts,
                }).to_csv(pred_csv, index=False)
                print(f"[OK] saved: {pred_csv}")

                # 2) 写唯一一行指标到 <dataset>.csv（覆盖写）
                ds_name = self.args.get("dataset_name", "dataset")
                met_csv = os.path.join(self.args["result_dir"], f"{ds_name}.csv")
                cols = [
                    "task_name",
                    "val_BLEU_1","val_BLEU_2","val_BLEU_3","val_BLEU_4",
                    "val_METEOR","val_ROUGE_L","val_CIDEr",
                    "test_BLEU_1","test_BLEU_2","test_BLEU_3","test_BLEU_4",
                    "test_METEOR","test_ROUGE_L","test_CIDEr"
                ]
                row = {k: math.nan for k in cols}
                row["task_name"] = self.args.get("task_name", f"run_{epoch}")

                # 从 monitor 里抓数值填充
                for k in cols:
                    if k == "task_name": 
                        continue
                    v = self.monitor.name2val.get(k, None)
                    if v is not None:
                        row[k] = float(v)

                pd.DataFrame([row], columns=cols).to_csv(met_csv, index=False)
        except Exception as e:
            print(f"[WARN] save best-only CSV failed: {e}")
        
        if self.args['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.monitor.dumpkv(epoch)
        with torch.no_grad():
            if (epoch % self._vis_every) == 0:
                try:
                    export_epoch_visuals(self, epoch)        # CHI / 回向一致性 / 预测文本
                except Exception as e:
                    print(f"[WARN] export epoch visuals failed: {e}")
                try:
                    export_x_token_attn_maps(self, epoch)    # 病理 token → patch 注意力（门控开/关）
                except Exception as e:
                    print(f"[WARN] export x-token attn failed: {e}")
        # （可选）只有想要“梯度型显著图”时才开：每 vis_grad_every 个 epoch 导一次
        try:
            if self._vis_grad_every > 0 and (epoch % self._vis_grad_every) == 0:
                export_grad_visuals(self, epoch, max_samples=2)
        except Exception as e:
            print(f"[WARN] export grad visuals failed: {e}")
        return log

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, img_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                # 收集 study_id（images_id 是 tuple/list）
                if isinstance(images_id, (list, tuple)):
                    img_ids.extend(list(images_id))
                else:
                    img_ids.append(images_id)

                test_res.extend(reports)
                test_gts.extend(ground_truths)

                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]",
                      end='', flush=True)

            assert len(img_ids) == len(test_res) == len(test_gts), \
                f"Length mismatch: ids={len(img_ids)}, pred={len(test_res)}, gt={len(test_gts)}"

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            save_report(test_res, test_gts, img_ids, os.path.join(self.checkpoint_dir, 'pred_gt_test.csv'))
            print("\n", test_met)

    def local_feature(self, idx):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p, img_ids = [], [], [], []
            hvs, hvs_id = [], []
            p = torch.zeros([1, self.args.max_seq_length]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if idx in images_id:
                    _, m = self.model(images, mode='extract')
                    attn = m['attn']

                    encoder_attn = encoder_heatmap(images, attn)
                    for i in range(encoder_attn.size(0)):
                        if idx == images_id[i]:
                            vis_heatmap(images[i], encoder_attn[i], title='encoder_attn')
                            print(images_id['i'])

    def keyword_feature(self, idx):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if idx in images_id:
                    key_id = self.model.tokenizer.token2idx['pleural']
                    target = torch.tensor([0, key_id, 0]).to(images).long()
                    target = target.unsqueeze(0).repeat(images.size(0), 1)
                    attn = self.model(images, target, mode='keyword')

                    encoder_attn = decoder_heatmap(images, attn)
                    for i in range(encoder_attn.size(0)):
                        if idx == images_id[i]:
                            vis_heatmap(images[i], encoder_attn[i], title='encoder_attn')
                            print(images_id[i])

    def extract(self):
        self.model.eval()
        with torch.no_grad():
            hvs, hvs_id = [], []
            p = torch.zeros([1, self.args.max_seq_length]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                h, _ = self.model(images, mode='extract')
                _hvs, _hvs_id = self.select_feature(h, images_id)
                hvs.extend(_hvs)
                hvs_id.extend(_hvs_id)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            torch.save({'id': hvs_id, 'feature': hvs}, 'results/mimic_cxr/DMIRG/DMIRG/118_feature.npy')

    def select_feature(self, h, images_id):
        hv_list = []
        hv_ids = []
        h = h.detach().cpu().numpy()
        for i in range(h.shape[0]):
            if images_id[i] in self.for_tsne:
                hv_list.append(h[i])
                hv_ids.append(images_id[i])
        return hv_list, hv_ids

def count_p(p):
    t = torch.unique(p, dim=0)
    l = t.size(0)
    return t, l

def mark_local(img, inx):
    """
    img [B, [N], 3, H, W]
    output [B, 3, H, W]
    """
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)
        for i in range(B):
            mask[i, inx[i, :]] = 1
        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)
    else:
        B, C, H, W = img.size()
        img = patchify(img, 16)
        mask = torch.zeros([B, 14 * 14]).to(img.device)
        for i in range(B):
            mask[i, inx[i, :]] = 1
        output = unpatchify(img, mask)
    return output

def encoder_heatmap(img, attn):
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)
    else:
        # B, C, H, W = img.size()
        mask = torch.zeros_like(img).to(img)
        mask = patchify(mask, 16)  # B, L, N

        length = len(attn)
        last_map = attn[0]
        for i in range(1, length):
            last_map = torch.matmul(attn[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        attn_map = last_map.mean(dim=1).unsqueeze(-1)
        mask += attn_map
        output = unpatchify(mask)
    return output

def decoder_heatmap(img, attn):
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)
    else:
        attn = attn[0] + attn[1] + attn[2]
        # B, C, H, W = img.size()
        mask = torch.zeros_like(img).to(img)
        mask = patchify(mask, 16)  # B, L, N
        last_map = attn[:, :, 1, 1:]
        attn_map = last_map.mean(dim=1).unsqueeze(-1)
        mask += attn_map
        output = unpatchify(mask)
    return output

def save_report(inference, reference, ids, output_csv):
    import pandas as pd
    sub_ids, stu_ids = [], []
    for x in ids:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            sub_ids.append(int(x[0]))
            stu_ids.append(int(x[1]))
        else:
            sub_ids.append(-1)
            stu_ids.append(-1)

    df = pd.DataFrame({
        "subject_id": sub_ids,
        "study_id": stu_ids,
        "pred_report": inference,
        "gt_report": reference
    })
    df.to_csv(output_csv, index=False)

"""
optimizers and lr scheduler
"""
import torch
import math


def build_optimizer(args, model):
    optim_name   = args["optim"]
    weight_decay = args["weight_decay"]
    amsgrad      = bool(args.get("amsgrad", True))
    task         = args.get("task", "train")

    lr_en  = float(args["lr_en"])
    lr_de  = float(args["lr_de"])
    lr_new = float(args.get("lr_new", lr_de * 2.5))

    # 关键词规则（先判“新模块”，再判“视觉侧”）
    enc_keywords = ("vis_embed", "encoder", "local_sample",  "global_sample",  "vdm", "lgfm", "unet", "vit", )
    new_keywords = ("dyce", "chi",  "cem", "conf",  "mem_proj", "mem_norm","seed_mlp", "feat",)

    enc_params, new_params, base_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name = n.lower()
        if any(k in name for k in new_keywords):
            new_params.append(p)
        elif any(k in name for k in enc_keywords):
            enc_params.append(p)
        else:
            base_params.append(p)

    # train/finetune：三组参数；其他：单组兜底
    if task in ("train", "finetune"):
        param_groups = []
        if len(enc_params) > 0:
            # 视觉侧加 lr_scale，供 WarmupAndSteplr 识别
            param_groups.append({"params": enc_params, "lr": lr_en,  "weight_decay": weight_decay, "lr_scale": 1.0})
        if len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": lr_de,  "weight_decay": weight_decay})
        if len(new_params) > 0:
            param_groups.append({"params": new_params, "lr": lr_new, "weight_decay": weight_decay})
        if len(param_groups) == 0:
            # 极端兜底（都被冻结）
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                             "lr": lr_de, "weight_decay": weight_decay}]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": lr_de, "weight_decay": weight_decay}]

    Optim = getattr(torch.optim, optim_name)
    optimizer = Optim(param_groups, weight_decay=weight_decay, amsgrad=amsgrad)

    # 可选：打印一下各组规模，方便核对
    try:
        print(f"[Optimizer] enc={sum(p.numel() for p in enc_params)}, "
              f"base={sum(p.numel() for p in base_params)}, "
              f"new={sum(p.numel() for p in new_params)} | "
              f"lr_en={lr_en}, lr_de={lr_de}, lr_new={lr_new}")
    except Exception:
        pass

    return optimizer

def build_lr_scheduler(args, optimizer, l):
    lr_scheduler_fn = args["lr_scheduler"]
    step_size = args["step_size"]
    epochs = args["epochs"]
    if lr_scheduler_fn == 'warmup_steplr':
        lr_scheduler = WarmupAndSteplr(optimizer, step_size * l, epochs * l)
    elif lr_scheduler_fn == 'warmup':
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, step_size * l, epochs * l)
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, args["lr_scheduler"])(optimizer, step_size, args["gamma"])
    print(f"Build {lr_scheduler_fn} for {args['optim']} in {args['task']}")
    return lr_scheduler

class WarmupAndSteplr(object):
    """
    - 视觉侧(带 lr_scale 的组)：线性 warmup -> cosine 衰减（以该组 base_lr 为基准）
    - 其他全部组：前半程保持 base_lr，训练进行到 50% 时整体降 10 倍（保持你原有策略）
    支持任意数量的 param_groups，不再假定“第0组=视觉侧、最后一组=decoder”
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.end_lr = 0.001
        self.current_step = 0
        self.optim = optimizer
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        # 记录每个 param_group 的初始 lr，作为该组的 base_lr
        self.base_lrs = [g.get("lr", 0.0) for g in self.optim.param_groups]

    def step(self):
        import math
        self.current_step += 1

        # 视觉侧 warmup + cosine 系数（无量纲）
        if self.current_step < self.num_warmup_steps:
            k_vis = float(self.current_step) / float(max(1, self.num_warmup_steps))
        else:
            progress = float(self.current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps))
            cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
            k_vis = max(0.0, cosine_lr * (1 - self.end_lr) + self.end_lr)

        # 非视觉侧中期衰减倍率
        k_dec = 0.1 if self.current_step == self.num_training_steps // 2 else 1.0

        # 逐组设置 lr
        last_vis_lr = 0.0
        last_dec_lr = 0.0
        for i, g in enumerate(self.optim.param_groups):
            base_lr_i = self.base_lrs[i]
            if "lr_scale" in g:
                # 视觉侧：warmup+cosine，考虑组内 lr_scale
                scale = g.get("lr_scale", 1.0)
                g["lr"] = k_vis * scale * base_lr_i
                last_vis_lr = g["lr"]
            else:
                # 其他全部：按 base_lr_i，在中点降 10 倍
                g["lr"] = k_dec * base_lr_i
                last_dec_lr = g["lr"]

        return last_vis_lr, last_dec_lr

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    when current_step < num_warmup_steps，
    new_lr =current_step/num_warmup_steps * base_lr
    when current_step >= num_warmup_steps，
    new_lr =(num_training_steps - current_step) / (num_training_steps -num_warmup_steps) * base_lr

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_line(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def lr_cosine(current_step: int):
        # linear warmup
        end_lr = 0.001
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine annealing decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        # lr = max(0.0, cosine_lr * (base_lr - end_lr) + end_lr)
        lr = max(0.0, cosine_lr * (1 - end_lr) + end_lr)
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_cosine, last_epoch)


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, base_lr=1e-3):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    try:
        num_layers = len(model.blocks) + 1
    except:
        num_layers = len(model.layers) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": base_lr,
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": base_lr,
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('layers'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

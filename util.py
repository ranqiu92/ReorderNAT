
import random
import numpy as np

import torch
import torch.nn as nn


class Transformer_LR_Schedule():

    def __init__(self, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step += 1
        scale = self.model_size ** -0.5
        scale *= min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return scale


class Linear_LR_Schedule():

    def __init__(self, initial_lr, final_lr, total_steps):
        self.initial_lr = initial_lr
        self.slope = (initial_lr - final_lr) / total_steps

    def __call__(self, step):
        scale = 1.0 - step * self.slope / self.initial_lr
        scale = max(scale, 0.)
        return scale


def set_random_seed(seed, is_cuda):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        torch.cuda.manual_seed(seed)

    return seed


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def mean(input, input_len=None):
    if input_len is not None:
        max_len = input.size(1)
        mask = ~sequence_mask(input_len, max_len).to(input.device)
        masked_input = input.masked_fill(mask.unsqueeze(-1), 0)
        input_sum = torch.sum(masked_input, dim=1)
        input_mean = input_sum / input_len.unsqueeze(-1).float()
        return input_mean
    else:
        return torch.mean(input, dim=1)

import math
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, dim, num_heads, peft_config):
        super().__init__()
        self.down_size = peft_config.lora_bottleneck
        self.lora_drop = nn.Dropout(p=0.1)
        self.num_heads = num_heads
        self.lora_a = nn.Linear(dim, self.down_size, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        self.lora_b = nn.Linear(self.down_size, dim * 3, bias=False)
        nn.init.zeros_(self.lora_b.weight)
        self.peft_config = peft_config
        self.merge_factor = 1.0

    def forward(self, x, q, k, v, B, N, C):
        qkv_delta = self.lora_a(self.lora_drop(x))

        qkv_delta = self.lora_b(qkv_delta).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q, v, k = q + self.merge_factor * q_delta, v + self.merge_factor *  v_delta, k + self.merge_factor *  k_delta
        return q, k, v

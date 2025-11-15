"""
Quick test script for FLaGPE specific functionality.
测试 FLaGPE 特定功能的快速脚本。
"""

import torch
from graphgps.encoder.fragmentation import get_fragmenter
from graphgps.encoder.flagpe_encoder import FLaGPENodeEncoder
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg, set_cfg

print("="*60)
print("FLaGPE 功能测试")
print("="*60)

# 1. 测试片段化
print("\n1. 测试片段化功能...")
edge_index = torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4, 4, 5],
                           [1, 0, 2, 1, 0, 2, 4, 3, 5, 4]], dtype=torch.long)
num_nodes = 6

for scheme in ['brics', 'ringspaths', 'random', 'single']:
    fragmenter = get_fragmenter(scheme)
    fragment_ids = fragmenter(edge_index, num_nodes)
    print(f"   {scheme:12s}: {fragment_ids.tolist()}")

# 2. 测试编码器
print("\n2. 测试 FLaGPE 编码器...")
set_cfg(cfg)
cfg.share.dim_in = 28

encoder = FLaGPENodeEncoder(dim_emb=64, expand_x=True)
print(f"   参数数量: {sum(p.numel() for p in encoder.parameters())}")
print(f"   Alpha 参数: {[f'{a.item():.3f}' for a in encoder.alpha]}")
print(f"   Beta 参数: {[f'{b.item():.3f}' for b in encoder.beta]}")

# 3. 测试 Forward Pass
print("\n3. 测试 Forward Pass...")
x = torch.randn(num_nodes, cfg.share.dim_in)
batch = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

try:
    batch_out = encoder(batch)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {batch_out.x.shape}")
    print(f"   ✅ Forward pass 成功!")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 4. 测试批处理
print("\n4. 测试批处理图...")
batch_list = []
for i in range(3):
    x_i = torch.randn(num_nodes, cfg.share.dim_in)
    batch_i = Data(x=x_i, edge_index=edge_index, num_nodes=num_nodes)
    batch_list.append(batch_i)

from torch_geometric.data import Batch
batched = Batch.from_data_list(batch_list)

try:
    batched_out = encoder(batched)
    print(f"   批处理输入: {batched.x.shape}")
    print(f"   批处理输出: {batched_out.x.shape}")
    print(f"   ✅ 批处理成功!")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 5. 测试不同参数设置
print("\n5. 测试不同超参数...")
configs = [
    {'k_hop': 3, 'dim_pe': 16},
    {'k_hop': 5, 'dim_pe': 32},
    {'k_hop': 10, 'dim_pe': 64},
]

for config in configs:
    cfg.posenc_FLaGPE.k_hop = config['k_hop']
    cfg.posenc_FLaGPE.dim_pe = config['dim_pe']
    try:
        enc = FLaGPENodeEncoder(dim_emb=64, expand_x=True)
        batch_test = Data(x=torch.randn(5, 28), 
                         edge_index=torch.randint(0, 5, (2, 10)),
                         num_nodes=5)
        out = enc(batch_test)
        print(f"   k_hop={config['k_hop']}, dim_pe={config['dim_pe']}: ✅")
    except Exception as e:
        print(f"   k_hop={config['k_hop']}, dim_pe={config['dim_pe']}: ❌ {e}")

print("\n" + "="*60)
print("测试完成！")
print("="*60)

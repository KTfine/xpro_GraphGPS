import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.graphgym.config import cfg

from graphgps.utils import negate_edge_index


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, use_bias, dim_edge_pe=0):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph
        self.dim_edge_pe = dim_edge_pe

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
      
        edge_in_dim = in_dim + dim_edge_pe if dim_edge_pe > 0 else in_dim
        self.E = nn.Linear(edge_in_dim, out_dim * num_heads, bias=use_bias)

        if self.full_graph:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            # 假边也需要调整输入维度（虽然假边PE为0）
            self.E_2 = nn.Linear(edge_in_dim, out_dim * num_heads, bias=use_bias)
            self.fake_edge_emb = fake_edge_emb

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[fake_edge_index[0]]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[fake_edge_index[1]]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1
            score_2 = torch.exp(score_2.sum(-1, keepdim=True).clamp(-5, 5))  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce='add')
        if self.full_graph:
            scatter(score_2, fake_edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        
        # ===== 实现 e'_ij = e_ij ∥ w_ij =====
        if self.dim_edge_pe > 0 and hasattr(batch, 'p_hat_modulated'):
            # 提取边对PE：从 p_hat_modulated[src, dst, :] 获取每条边的 w_ij
            edge_pe = self._extract_edge_pe(batch)  # [num_edges, dim_edge_pe]
            # 拼接边特征和边PE
            edge_attr_with_pe = torch.cat([batch.edge_attr, edge_pe], dim=-1)  # [num_edges, edge_dim + dim_pe]
            E = self.E(edge_attr_with_pe)
        else:
            # 检查维度是否匹配
            if self.dim_edge_pe > 0:
                raise ValueError(
                    f"dim_edge_pe={self.dim_edge_pe} > 0 但缺少 batch.p_hat_modulated。"
                    "请确保 SANLayer.forward() 在调用 attention 前保存了 p_hat_modulated。"
                )
            E = self.E(batch.edge_attr)

        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            # 假边需要拼接零PE以匹配维度
            if self.dim_edge_pe > 0:
                zero_pe = torch.zeros(1, self.dim_edge_pe, device=dummy_edge.device, dtype=dummy_edge.dtype)
                dummy_edge_with_pe = torch.cat([dummy_edge, zero_pe], dim=-1)
                E_2 = self.E_2(dummy_edge_with_pe)
            else:
                E_2 = self.E_2(dummy_edge)

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out

    def _extract_edge_pe(self, batch):
        """
        从 p_hat_modulated 中提取每条边的边对PE w_ij
        
        对于边 (src, dst)，提取 p_hat_modulated[src, dst, :]
        
        Args:
            batch: 包含 edge_index 和 p_hat_modulated 的 batch 对象
            
        Returns:
            edge_pe: [num_edges, dim_edge_pe] 每条边的PE
        """
        edge_index = batch.edge_index  # [2, num_edges]
        num_edges = edge_index.size(1)
        
        # 检查是单图还是批处理
        if not hasattr(batch, 'ptr') or batch.ptr is None or len(batch.ptr) <= 2:
            # 单图情况
            p_hat_mod = batch.p_hat_modulated[0]  # [N, N, dim_pe]
            src_nodes = edge_index[0]  # [num_edges]
            dst_nodes = edge_index[1]  # [num_edges]
            # 提取每条边的PE
            edge_pe = p_hat_mod[src_nodes, dst_nodes, :]  # [num_edges, dim_pe]
        else:
            # 批处理：向量化提取，避免逐边循环
            batch_indices = batch.batch  # [total_nodes] 每个节点所属的图索引
            batch_ptr = batch.ptr  # [batch_size + 1]
            
            src_global = edge_index[0]  # [num_edges]
            dst_global = edge_index[1]  # [num_edges]
            
            # 获取每条边所属的图索引 [num_edges]
            graph_idx_per_edge = batch_indices[src_global]
            
            # 获取每条边的节点偏移 [num_edges]
            node_offsets = batch_ptr[graph_idx_per_edge]
            
            # 计算局部索引 [num_edges]
            src_local = src_global - node_offsets
            dst_local = dst_global - node_offsets
            
            # 分组提取：为每个图的边收集PE
            edge_pe_list = []
            for graph_idx in range(len(batch_ptr) - 1):
                # 找出属于这个图的边
                mask = graph_idx_per_edge == graph_idx
                if not mask.any():
                    continue
                
                # 获取这些边的局部索引
                src_local_i = src_local[mask]
                dst_local_i = dst_local[mask]
                
                # 从该图的 p_hat_modulated 中一次性提取所有边的PE
                p_hat_mod_i = batch.p_hat_modulated[graph_idx]  # [N_i, N_i, dim_pe]
                edge_pe_i = p_hat_mod_i[src_local_i, dst_local_i, :]  # [num_edges_i, dim_pe]
                edge_pe_list.append(edge_pe_i)
            
            # 拼接所有图的边PE
            edge_pe = torch.cat(edge_pe_list, dim=0)  # [num_edges, dim_pe]
        
        return edge_pe


class SANLayer(nn.Module):
    """GraphTransformerLayer from SAN with FLaGPE support.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    
    Extended with Fragment-aware Random Walk layerwise PE fusion.
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False, layer_idx=0):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.layer_idx = layer_idx  # Layer index for FLaGPE
        
        # Check if FLaGPE is enabled
        self.use_flagpe = (hasattr(cfg, 'posenc_FLaGPE') and 
                          cfg.posenc_FLaGPE.enable and 
                          cfg.posenc_FLaGPE.pass_as_var)
        
        if self.use_flagpe:
            self.k_hop = cfg.posenc_FLaGPE.k_hop  # 从配置读取k_hop
            self.dim_pe = self.k_hop  # PE维度等于k_hop
            self.dim_features = in_dim - self.dim_pe
            
            # α^(ℓ) ∈ [0,1]：控制片段内vs片段间的权重
            self.alpha = nn.Parameter(torch.tensor(cfg.posenc_FLaGPE.alpha_init))
            # β^(ℓ) ∈ [0,1]：控制新旧PE的融合比例
            self.beta = nn.Parameter(torch.tensor(cfg.posenc_FLaGPE.beta_init))
        else:
            self.dim_pe = 0
            self.dim_features = in_dim
        
        # 传递 dim_edge_pe 给 attention layer 以实现 e'_ij = e_ij ∥ w_ij
        self.attention = MultiHeadAttentionLayer(gamma=gamma,
                                                 in_dim=in_dim,
                                                 out_dim=out_dim // num_heads,
                                                 num_heads=num_heads,
                                                 full_graph=full_graph,
                                                 fake_edge_emb=fake_edge_emb,
                                                 use_bias=use_bias,
                                                 dim_edge_pe=self.dim_pe if self.use_flagpe else 0)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def compute_modulated_pe(self, batch):
        """
        Step 3-4 (按论文规范)：
        对预计算的RRWP w_ij 应用层级特定的fragment-aware加权
        
        公式：
        P̃_ij^(t,ℓ) = [α^(ℓ)·(1-δ_ij) + (1-α^(ℓ))·δ_ij] · P_ij^(t)
                      / Σ_j' [α^(ℓ)·(1-δ_ij') + (1-α^(ℓ))·δ_ij'] · P_ij'^(t)
        
        然后投影：p̂_ij^(ℓ) = MLP(P̃_ij^(ℓ))
        
        Returns:
            p_hat_modulated: [num_nodes, num_nodes, K] 修改后的PE
        """
        # 检查必要的预计算数据
        if not hasattr(batch, 'w_rrwp') or not hasattr(batch, 'delta_ij') or not hasattr(batch, 'dim_pe'):
            raise ValueError("FLaGPE预计算数据缺失！检查Encoder是否使用FLaGPE")
        
        device = batch.x.device
        batch_size = batch.batch_size
        
        # 获取层级特定的alpha参数
        alpha = torch.sigmoid(self.alpha)  # 标量，∈(0,1)
        
        # 获取PE维度
        dim_pe = batch.dim_pe
        
        # 处理单图或批处理
        if batch_size == 1:
            w_rrwp = batch.w_rrwp[0]  # [N, N, K]
            delta_ij = batch.delta_ij[0]  # [N, N] 
            
            p_hat_modulated = self._apply_alpha_modulation(
                w_rrwp, delta_ij, alpha, dim_pe
            )
            p_hat_modulated_list = [p_hat_modulated]
        else:
            # 批处理：逐图处理
            p_hat_modulated_list = []
            for i in range(batch_size):
                w_rrwp_i = batch.w_rrwp[i]
                delta_ij_i = batch.delta_ij[i]  
                
                p_hat_mod_i = self._apply_alpha_modulation(
                    w_rrwp_i, delta_ij_i, alpha, dim_pe
                )
                p_hat_modulated_list.append(p_hat_mod_i)
        
        return p_hat_modulated_list
    
    def _apply_alpha_modulation(self, w_rrwp, delta_ij, alpha, dim_pe):
        """
        对单个图应用alpha加权和归一化
        
        Args:
            w_rrwp: [N, N, K] 原始RRWP
            delta_ij: [N, N] 预计算的片段掩码（Encoder已计算）
            alpha: 标量，层级特定的参数
            dim_pe: PE维度（等于K）
            
        Returns:
            p_hat_modulated: [N, N, dim_pe] 加权后的PE（直接是加权后RRWP）
        """
        num_nodes = w_rrwp.size(0)
        
        # Step 3a: 使用预计算的片段掩码 δ_ij
        delta = delta_ij
        
        # Step 3b: 计算权重系数
        weight_coeff = alpha * (1 - delta) + (1 - alpha) * delta  # [N, N]
        
        # Step 3c: 对所有K步一次性应用加权和归一化
        weight_coeff_expanded = weight_coeff.unsqueeze(-1)  # [N, N] -> [N, N, 1]

        numerator = weight_coeff_expanded * w_rrwp  # [N, N, K]
        
        denominator = numerator.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [N, 1, K]
        w_rrwp_modulated = numerator / denominator  # [N, N, K]
        
        return w_rrwp_modulated

    def forward(self, batch):
        h = batch.x
        
        # ===== Step 3-5：Fragment-aware PE模块化和融合 =====
        if self.use_flagpe and hasattr(batch, 'w_rrwp'):

            h_features = h[:, :self.dim_features]  # [N, dim_features]
            h_pe_old = h[:, self.dim_features:]    # [N, dim_pe]（前一层的PE）
            
            # Step 3-4: 对预计算的RRWP应用alpha加权，得到修改的PE
            p_hat_modulated_list = self.compute_modulated_pe(batch)
            
            # Step 4: 
            # 使用新矩阵的对角线元素作为节点PE  
            if batch.batch_size == 1:
                p_hat_mod = p_hat_modulated_list[0]  # [N, N, dim_pe]
                p_hat_l = p_hat_mod.diagonal(dim1=0, dim2=1).T  # [N, dim_pe]
            else:
                # 批处理
                p_hat_l_list = []
                start_idx = 0
                for i, p_hat_mod_i in enumerate(p_hat_modulated_list):
                    num_nodes_i = p_hat_mod_i.size(0)
                    p_hat_i = p_hat_mod_i.diagonal(dim1=0, dim2=1).T  # [N_i, dim_pe]
                    p_hat_l_list.append(p_hat_i)
                p_hat_l = torch.cat(p_hat_l_list, dim=0)
            
            # Step 5: 分层PE融合
            beta = torch.sigmoid(self.beta)  # ∈(0,1)
            h_pe_new = beta * p_hat_l + (1 - beta) * h_pe_old
            
            # 重新拼接特征和更新的PE
            h = torch.cat([h_features, h_pe_new], dim=-1)
            batch.x = h  # 更新batch.x
            
            # 为下一层保存当前的PE
            batch.pe_current = h_pe_new
            
            # 保存 p_hat_modulated 供 attention layer 提取边对PE =====
            batch.p_hat_modulated = p_hat_modulated_list
        
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)

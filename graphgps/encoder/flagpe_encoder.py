import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj

from graphgps.encoder.fragmentation import get_fragmenter


@register_node_encoder('FLaGPE')
class FLaGPENodeEncoder(torch.nn.Module):
    
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        
        dim_in = cfg.share.dim_in
        
        pecfg = cfg.posenc_FLaGPE
        self.k_hop = pecfg.k_hop  # 随机游走步数K，也是PE维度
        self.fragment_scheme = pecfg.fragment_scheme
        
        # PE维度 = 随机游走步数
        self.dim_pe = pecfg.dim_pe
        
        if dim_emb - self.dim_pe < 0:
            raise ValueError(f"PE dim {self.dim_pe} (k_hop={self.k_hop}) too large for embedding {dim_emb}")
        
        # 节点特征投影
        if expand_x and dim_emb - self.dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - self.dim_pe)
        self.expand_x = expand_x and dim_emb - self.dim_pe > 0
        
        # 初始化分割器
        self.fragmenter = get_fragmenter(self.fragment_scheme)
        
    def compute_transition_matrices(self, edge_index, num_nodes, device):
        """
        Step 1: 预计算k步转移矩阵 P(t) = P^t
        
        Args:
            edge_index: [2, num_edges]
            num_nodes: 节点数
            device: torch device
            
        Returns:
            P_list: 列表，P_list[t] = [num_nodes, num_nodes]，t=0,1,...,K-1
        """
        # 构造邻接矩阵
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N, N]
        
        # 计算度矩阵和转移矩阵
        degree = adj.sum(dim=1).clamp(min=1.0)
        D_inv = torch.diag(1.0 / degree)  # D^{-1}
        P = torch.matmul(D_inv, adj)  # P = D^{-1}A
        
        # 预计算P^t，t=0,1,...,K-1
        P_list = [torch.eye(num_nodes, device=device)]  # P^0 = I
        P_power = P.clone()
        
        for t in range(1, self.k_hop):
            P_list.append(P_power.clone())
            P_power = torch.matmul(P_power, P)
        
        return P_list
    
    def compute_rrwp(self, P_list, num_nodes):
        """
        Step 2-3: 计算RRWP向量 w_ij = [P_ij(0), P_ij(1), ..., P_ij(K-1)]
        
        边对(i,j)级的PE（GRIT风格）
        
        Args:
            P_list: 转移矩阵列表
            num_nodes: 节点数
            
        Returns:
            w_rrwp: [num_nodes, num_nodes, k_hop] 张量
                   w_rrwp[i,j,:] = [P_ij(0), P_ij(1), ..., P_ij(K-1)]
        """
        K = len(P_list)
        w_rrwp = torch.stack(P_list, dim=-1)  # [N, N, K]
        return w_rrwp
    
    def forward(self, batch):
        """
        前向传播：预计算所有需要的矩阵和向量
        
        步骤：
        1. 计算P(t)矩阵
        2. 计算RRWP w_ij（直接作为PE，无需MLP投影）
        3. 获取片段ID和片段掩码
        4. 初始化节点特征
        5. 存储所有信息到batch供Layer使用
        """
        device = batch.edge_index.device
        
        # ===== 处理单个图或批处理 =====
        if not hasattr(batch, 'ptr'):
            # 单个图
            num_nodes = batch.num_nodes
            edge_index = batch.edge_index
            
            # Step 1: 计算转移矩阵 P(t)
            P_list = self.compute_transition_matrices(edge_index, num_nodes, device)
            
            # Step 2-3: 计算RRWP w_ij [N, N, K]
            w_rrwp = self.compute_rrwp(P_list, num_nodes)
            
            # 获取片段ID
            smiles = batch.smiles[0] if hasattr(batch, 'smiles') else None
            frag_id = self.fragmenter(edge_index, num_nodes, smiles=smiles)
            frag_id = frag_id.to(device)
            
            # ===== 预计算片段掩码 δ_ij（所有层共用）=====
            frag_matrix = frag_id.unsqueeze(1) == frag_id.unsqueeze(0)  # [N, N]
            delta_ij = frag_matrix.float()  # 同片段为1，不同片段为0
            
            w_rrwp_all = [w_rrwp]
            delta_ij_all = [delta_ij]
            
        else:
            # 批处理：每个图分别处理
            w_rrwp_all = []
            delta_ij_all = []
            
            for i in range(len(batch.ptr) - 1):
                start_idx = batch.ptr[i].item()
                end_idx = batch.ptr[i + 1].item()
                num_nodes_i = end_idx - start_idx
                
                # 提取子图边
                node_mask = (batch.batch == i)
                edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
                edge_index_i = batch.edge_index[:, edge_mask] - start_idx
                
                # 计算转移矩阵
                P_list_i = self.compute_transition_matrices(edge_index_i, num_nodes_i, device)
                
                # 计算RRWP
                w_rrwp_i = self.compute_rrwp(P_list_i, num_nodes_i)
                
                # 获取片段ID
                smiles_i = batch.smiles[i] if hasattr(batch, 'smiles') else None
                frag_id_i = self.fragmenter(edge_index_i, num_nodes_i, smiles=smiles_i)
                frag_id_i = frag_id_i.to(device)
                
                # ===== 预计算片段掩码 δ_ij =====
                frag_matrix_i = frag_id_i.unsqueeze(1) == frag_id_i.unsqueeze(0)
                delta_ij_i = frag_matrix_i.float()
                
                w_rrwp_all.append(w_rrwp_i)
                delta_ij_all.append(delta_ij_i)
        
        # ===== 初始化节点特征 =====
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        # 初始PE：零向量（Layer会在第一次前向传播时计算实际PE）
        num_nodes_total = h.size(0)
        p_init = torch.zeros(num_nodes_total, self.dim_pe, device=device)
        
        # 拼接特征和PE
        batch.x = torch.cat([h, p_init], dim=1)
        
        # ===== 存储供Layer使用 =====
        batch.w_rrwp = w_rrwp_all  # RRWP w_ij [N,N,K]，直接作为PE使用
        batch.delta_ij = delta_ij_all  # 预计算的片段掩码（避免每层重复计算）
        batch.dim_pe = self.dim_pe  # PE维度（等于k_hop）
        batch.batch_size = len(batch.ptr) - 1 if hasattr(batch, 'ptr') else 1
        
        return batch

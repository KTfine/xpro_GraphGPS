"""
FLaGPE: Fragment-aware Layerwise Graph Positional Encoding

This module implements fragment-aware positional encoding that combines:
1. Intra-fragment random walk statistics (local structure)
2. Inter-fragment random walk statistics (global context)
3. Layerwise learnable fusion parameters (α and β)
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

# Import fragmenter from the same encoder directory
from graphgps.encoder.fragmentation import get_fragmenter


@register_node_encoder('FLaGPE')
class FLaGPENodeEncoder(torch.nn.Module):
    """
    Fragment-aware Layerwise Graph Positional Encoding (FLaGPE).
    
    This encoder computes positional encodings based on random walk probabilities
    within and across molecular fragments, with layerwise adaptive fusion.
    
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """
    
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        
        dim_in = cfg.share.dim_in  # Expected original input node features dim
        
        # Load FLaGPE configuration
        pecfg = cfg.posenc_FLaGPE
        dim_pe = pecfg.dim_pe  # Size of PE embedding
        self.k_hop = pecfg.k_hop  # Number of random walk steps
        self.fragment_scheme = pecfg.fragment_scheme  # Fragmentation strategy
        self.mlp_hidden = pecfg.mlp_hidden  # Hidden dimension for MLP
        self.alpha_init = pecfg.alpha_init  # Initial value for alpha (inter-fragment weight)
        self.beta_init = pecfg.beta_init  # Initial value for beta (layer fusion weight)
        self.num_layers = pecfg.num_layers if hasattr(pecfg, 'num_layers') else 1
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable
        
        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                           f"desired embedding size of {dim_emb}.")
        
        # Linear projection for original node features
        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0
        
        # Initialize fragmenter
        self.fragmenter = get_fragmenter(self.fragment_scheme)
        
        # Learnable parameters for fragment fusion (per layer)
        # α controls intra vs inter-fragment weighting
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(self.alpha_init))
            for _ in range(self.num_layers)
        ])
        
        # β controls layer-wise PE fusion
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(self.beta_init))
            for _ in range(self.num_layers)
        ])
        
        # MLP for encoding random walk statistics
        # Input: k_hop random walk probabilities for each node pair
        self.rw_encoder = nn.Sequential(
            nn.Linear(self.k_hop, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, dim_pe),
            nn.ReLU()
        )
        
        # Optional: LayerNorm for stability
        self.layer_norm = nn.LayerNorm(dim_pe)
        
    def compute_random_walk_matrix(self, edge_index, num_nodes):
        """
        Compute k-step random walk probability matrices.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            rw_matrices: List of [num_nodes, num_nodes] tensors for each step
        """
        device = edge_index.device
        
        # Construct adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [num_nodes, num_nodes]
        
        # Compute degree matrix
        degree = adj.sum(dim=1).clamp(min=1)  # Avoid division by zero
        D_inv = torch.diag(1.0 / degree)
        
        # Transition matrix: P = D^{-1} A
        P = torch.matmul(D_inv, adj)
        
        # Compute powers of P: P^0, P^1, ..., P^k
        rw_matrices = [torch.eye(num_nodes, device=device)]  # P^0 = I
        P_current = P
        
        for _ in range(1, self.k_hop):
            rw_matrices.append(P_current.clone())
            P_current = torch.matmul(P_current, P)
        
        return rw_matrices
    
    def compute_fragment_aware_rw(self, rw_matrices, fragment_ids, layer_idx=0):
        """
        Compute fragment-aware random walk encoding.
        
        Args:
            rw_matrices: List of random walk matrices [num_nodes, num_nodes]
            fragment_ids: Fragment assignment for each node [num_nodes]
            layer_idx: Current layer index for alpha selection
            
        Returns:
            pe_matrix: Positional encoding matrix [num_nodes, num_nodes, k_hop]
        """
        num_nodes = fragment_ids.size(0)
        device = fragment_ids.device
        
        # Get alpha for current layer
        alpha = torch.sigmoid(self.alpha[min(layer_idx, len(self.alpha) - 1)])
        
        # Create masks for intra and inter-fragment pairs
        fragment_matrix = fragment_ids.unsqueeze(1) == fragment_ids.unsqueeze(0)  # [num_nodes, num_nodes]
        intra_mask = fragment_matrix.float()
        inter_mask = (~fragment_matrix).float()
        
        # Combine random walk matrices with fragment awareness
        pe_list = []
        for t, rw_t in enumerate(rw_matrices):
            # Weighted combination of intra and inter-fragment random walks
            pe_t = (1 - alpha) * (rw_t * intra_mask) + alpha * (rw_t * inter_mask)
            pe_list.append(pe_t.unsqueeze(-1))
        
        # Stack to get [num_nodes, num_nodes, k_hop]
        pe_matrix = torch.cat(pe_list, dim=-1)
        
        return pe_matrix
    
    def encode_pe(self, pe_matrix):
        """
        Encode the positional encoding matrix using MLP.
        
        Args:
            pe_matrix: [num_nodes, num_nodes, k_hop]
            
        Returns:
            node_pe: [num_nodes, dim_pe]
        """
        num_nodes = pe_matrix.size(0)
        
        # For each node, aggregate PE from all other nodes
        # Simple approach: mean pooling over neighbors
        # Shape: [num_nodes, k_hop]
        aggregated_pe = pe_matrix.mean(dim=1)
        
        # Encode through MLP
        node_pe = self.rw_encoder(aggregated_pe)  # [num_nodes, dim_pe]
        
        # Apply layer normalization
        node_pe = self.layer_norm(node_pe)
        
        return node_pe
    
    def forward(self, batch):
        """
        Forward pass to compute and attach FLaGPE to the batch.
        
        Args:
            batch: PyG Batch object
            
        Returns:
            batch: Modified batch with PE attached
        """
        device = batch.edge_index.device
        
        # Handle batched graphs
        if not hasattr(batch, 'ptr'):
            # Single graph
            num_nodes = batch.num_nodes
            edge_index = batch.edge_index
            
            # Compute fragment IDs
            smiles = batch.smiles[0] if hasattr(batch, 'smiles') else None
            fragment_ids = self.fragmenter(edge_index, num_nodes, smiles=smiles)
            fragment_ids = fragment_ids.to(device)
            
            # Compute random walk matrices
            rw_matrices = self.compute_random_walk_matrix(edge_index, num_nodes)
            
            # Compute fragment-aware PE
            pe_matrix = self.compute_fragment_aware_rw(rw_matrices, fragment_ids, layer_idx=0)
            
            # Encode to node-level PE
            pos_enc = self.encode_pe(pe_matrix)
            
        else:
            # Batched graphs
            pos_enc_list = []
            
            for i in range(len(batch.ptr) - 1):
                start_idx = batch.ptr[i].item()
                end_idx = batch.ptr[i + 1].item()
                num_nodes_i = end_idx - start_idx
                
                # Extract subgraph
                node_mask = (batch.batch == i)
                edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
                edge_index_i = batch.edge_index[:, edge_mask] - start_idx
                
                # Compute fragment IDs
                smiles_i = batch.smiles[i] if hasattr(batch, 'smiles') else None
                fragment_ids_i = self.fragmenter(edge_index_i, num_nodes_i, smiles=smiles_i)
                fragment_ids_i = fragment_ids_i.to(device)
                
                # Compute random walk matrices
                rw_matrices_i = self.compute_random_walk_matrix(edge_index_i, num_nodes_i)
                
                # Compute fragment-aware PE
                pe_matrix_i = self.compute_fragment_aware_rw(rw_matrices_i, fragment_ids_i, layer_idx=0)
                
                # Encode to node-level PE
                pos_enc_i = self.encode_pe(pe_matrix_i)
                pos_enc_list.append(pos_enc_i)
            
            pos_enc = torch.cat(pos_enc_list, dim=0)
        
        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), dim=1)
        
        # Keep PE also separate in a variable (e.g. for skip connections)
        if self.pass_as_var:
            batch.pe_FLaGPE = pos_enc
        
        return batch

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.san_layer import SANLayer
from graphgps.layer.san2_layer import SAN2Layer
from graphgps.layer.san_layer_flag import SANLayer as SANLayerFLaGPE


@register_network('SANTransformer')
class SANTransformer(torch.nn.Module):
    """Spectral Attention Network (SAN) Graph Transformer.
    https://arxiv.org/abs/2106.03893
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        fake_edge_emb = torch.nn.Embedding(1, cfg.gt.dim_hidden)
        # torch.nn.init.xavier_uniform_(fake_edge_emb.weight.data)
        
        # Select layer class based on FLaGPE configuration
        if hasattr(cfg, 'posenc_FLaGPE') and cfg.posenc_FLaGPE.enable:
            # Use FLaGPE-enabled layer
            layer_map = {
                'SANLayer': SANLayerFLaGPE,
                'SAN2Layer': SAN2Layer,
            }
        else:
            # Use standard layers
            layer_map = {
                'SANLayer': SANLayer,
                'SAN2Layer': SAN2Layer,
            }
        
        Layer = layer_map.get(cfg.gt.layer_type)
        layers = []
        for layer_idx in range(cfg.gt.layers):
            layer_kwargs = {
                'gamma': cfg.gt.gamma,
                'in_dim': cfg.gt.dim_hidden,
                'out_dim': cfg.gt.dim_hidden,
                'num_heads': cfg.gt.n_heads,
                'full_graph': cfg.gt.full_graph,
                'fake_edge_emb': fake_edge_emb,
                'dropout': cfg.gt.dropout,
                'layer_norm': cfg.gt.layer_norm,
                'batch_norm': cfg.gt.batch_norm,
                'residual': cfg.gt.residual
            }
            # Add layer_idx only when using FLaGPE-enabled layer
            if hasattr(cfg, 'posenc_FLaGPE') and cfg.posenc_FLaGPE.enable and cfg.gt.layer_type == 'SANLayer':
                layer_kwargs['layer_idx'] = layer_idx
            layers.append(Layer(**layer_kwargs))
        self.trf_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

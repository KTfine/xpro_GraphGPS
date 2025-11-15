"""
Verification script for FLaGPE implementation.
This script tests the configuration and basic functionality of FLaGPE.
"""

import torch
import sys
import os

# Add graphgps to path
sys.path.insert(0, os.path.abspath('.'))

def test_fragmentation():
    """Test fragmentation module."""
    print("=" * 60)
    print("Testing Fragmentation Module")
    print("=" * 60)
    
    from graphgps.encoder.fragmentation import get_fragmenter
    
    # Test different fragmenters
    fragmenters = ['brics', 'ringspaths', 'random', 'single']
    
    # Create a simple graph (triangle + path)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4],
                               [1, 0, 2, 1, 0, 2, 4, 3]], dtype=torch.long)
    num_nodes = 5
    
    for frag_name in fragmenters:
        try:
            fragmenter = get_fragmenter(frag_name)
            fragment_ids = fragmenter(edge_index, num_nodes)
            print(f"✓ {frag_name:15s} - Fragment IDs: {fragment_ids.tolist()}")
        except Exception as e:
            print(f"✗ {frag_name:15s} - Error: {e}")
    
    print()


def test_config():
    """Test configuration loading."""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    try:
        from torch_geometric.graphgym.config import cfg, set_cfg
        from graphgps.config.posenc_config import set_cfg_posenc
        
        # Initialize config
        set_cfg(cfg)
        
        # Check if FLaGPE config exists
        assert hasattr(cfg, 'posenc_FLaGPE'), "FLaGPE config not found!"
        
        print("✓ Configuration loaded successfully")
        print(f"  - dim_pe: {cfg.posenc_FLaGPE.dim_pe}")
        print(f"  - k_hop: {cfg.posenc_FLaGPE.k_hop}")
        print(f"  - fragment_scheme: {cfg.posenc_FLaGPE.fragment_scheme}")
        print(f"  - mlp_hidden: {cfg.posenc_FLaGPE.mlp_hidden}")
        print(f"  - alpha_init: {cfg.posenc_FLaGPE.alpha_init}")
        print(f"  - beta_init: {cfg.posenc_FLaGPE.beta_init}")
        print()
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_encoder_import():
    """Test if FLaGPE encoder can be imported."""
    print("=" * 60)
    print("Testing Encoder Import")
    print("=" * 60)
    
    try:
        from graphgps.encoder.flagpe_encoder import FLaGPENodeEncoder
        print("✓ FLaGPENodeEncoder imported successfully")
        print()
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_encoder_registration():
    """Test if FLaGPE is registered."""
    print("=" * 60)
    print("Testing Encoder Registration")
    print("=" * 60)
    
    try:
        from torch_geometric.graphgym.register import node_encoder_dict
        
        # Check if FLaGPE and combinations are registered
        encoders_to_check = [
            'FLaGPE',
            'Atom+FLaGPE',
            'TypeDictNode+FLaGPE',
            'LinearNode+FLaGPE'
        ]
        
        for enc_name in encoders_to_check:
            if enc_name in node_encoder_dict:
                print(f"✓ {enc_name:25s} - Registered")
            else:
                print(f"✗ {enc_name:25s} - Not found")
        
        print()
        
    except Exception as e:
        print(f"✗ Registration check error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_encoder_instantiation():
    """Test if encoder can be instantiated."""
    print("=" * 60)
    print("Testing Encoder Instantiation")
    print("=" * 60)
    
    try:
        from torch_geometric.graphgym.config import cfg, set_cfg
        from graphgps.encoder.flagpe_encoder import FLaGPENodeEncoder
        
        # Set minimal config
        set_cfg(cfg)
        cfg.share.dim_in = 28
        
        # Create encoder
        dim_emb = 64
        encoder = FLaGPENodeEncoder(dim_emb=dim_emb, expand_x=True)
        
        print("✓ Encoder instantiated successfully")
        print(f"  - Parameters: {sum(p.numel() for p in encoder.parameters())}")
        print(f"  - Alpha layers: {len(encoder.alpha)}")
        print(f"  - Beta layers: {len(encoder.beta)}")
        print()
        
    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    try:
        from torch_geometric.graphgym.config import cfg, set_cfg
        from graphgps.encoder.flagpe_encoder import FLaGPENodeEncoder
        from torch_geometric.data import Data
        
        # Set config
        set_cfg(cfg)
        cfg.share.dim_in = 28
        
        # Create encoder
        dim_emb = 64
        encoder = FLaGPENodeEncoder(dim_emb=dim_emb, expand_x=True)
        
        # Create dummy batch
        num_nodes = 10
        x = torch.randn(num_nodes, cfg.share.dim_in)
        edge_index = torch.randint(0, num_nodes, (2, 30))
        
        batch = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        # Forward pass
        batch_out = encoder(batch)
        
        print("✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {batch_out.x.shape}")
        print(f"  - Expected shape: ({num_nodes}, {dim_emb})")
        
        assert batch_out.x.shape == (num_nodes, dim_emb), "Output shape mismatch!"
        print("✓ Output shape correct")
        print()
        
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_config_file():
    """Test loading the YAML config file."""
    print("=" * 60)
    print("Testing YAML Configuration File")
    print("=" * 60)
    
    try:
        import yaml
        
        config_path = 'configs/GPS/zinc-GPS+FLaGPE.yaml'
        
        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            print()
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config file loaded: {config_path}")
        
        # Check key settings
        assert 'posenc_FLaGPE' in config, "posenc_FLaGPE not in config!"
        assert config['dataset']['node_encoder_name'] == 'TypeDictNode+FLaGPE', \
            "Encoder name mismatch!"
        
        print("  - Encoder: TypeDictNode+FLaGPE")
        print(f"  - FLaGPE settings:")
        for key, value in config['posenc_FLaGPE'].items():
            print(f"    - {key}: {value}")
        
        print("✓ Config file valid")
        print()
        
    except Exception as e:
        print(f"✗ Config file error: {e}")
        import traceback
        traceback.print_exc()
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FLaGPE Implementation Verification")
    print("=" * 60 + "\n")
    
    # Run tests
    test_fragmentation()
    test_config()
    test_encoder_import()
    test_encoder_registration()
    test_encoder_instantiation()
    test_forward_pass()
    test_config_file()
    
    print("=" * 60)
    print("Verification Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run training with: python main.py --cfg configs/GPS/zinc-GPS+FLaGPE.yaml")
    print("2. Monitor the training logs for any runtime errors")
    print("3. Compare results with baseline (zinc-GPS.yaml)")
    print()


if __name__ == '__main__':
    main()

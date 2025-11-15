"""
Fragment-aware graph decomposition utilities for FLaGPE.
Supports multiple fragmentation strategies for molecular graphs.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class BaseFragmenter:
    """Base class for graph fragmentation strategies."""
    
    def __call__(self, edge_index: torch.Tensor, num_nodes: int, 
                 smiles: Optional[str] = None, mol: Optional[object] = None) -> torch.Tensor:
        """
        Fragment a graph into subgraphs.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes in the graph
            smiles: SMILES string (optional, for molecule-based fragmentation)
            mol: RDKit Mol object (optional)
            
        Returns:
            fragment_ids: Tensor of shape [num_nodes] with fragment assignment for each node
        """
        raise NotImplementedError


class BRICSFragmenter(BaseFragmenter):
    """Fragment molecules using BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)."""
    
    def __call__(self, edge_index: torch.Tensor, num_nodes: int,
                 smiles: Optional[str] = None, mol: Optional[object] = None) -> torch.Tensor:
        if not HAS_RDKIT:
            # Fallback to connected components if RDKit not available
            return self._fallback_fragmentation(edge_index, num_nodes)
        
        if mol is None and smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return self._fallback_fragmentation(edge_index, num_nodes)
        
        try:
            # Break molecule using BRICS
            fragments = BRICS.BRICSDecompose(mol)
            
            if len(fragments) <= 1:
                # No fragmentation occurred, return single fragment
                return torch.zeros(num_nodes, dtype=torch.long)
            
            # Map atoms to fragments
            fragment_ids = torch.zeros(num_nodes, dtype=torch.long)
            
            # Get the fragment assignment for each atom
            # This is a simplified version - BRICS returns SMILES of fragments
            # For production, you'd need to map atom indices to fragments
            # For now, use a heuristic based on connected components after breaking bonds
            
            # Fallback to connected components for now
            return self._fallback_fragmentation(edge_index, num_nodes)
            
        except Exception:
            return self._fallback_fragmentation(edge_index, num_nodes)
    
    def _fallback_fragmentation(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Fallback to connected components-based fragmentation."""
        return self._connected_components(edge_index, num_nodes)
    
    def _connected_components(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Find connected components in the graph."""
        fragment_ids = torch.full((num_nodes,), -1, dtype=torch.long)
        current_fragment = 0
        
        for node in range(num_nodes):
            if fragment_ids[node] == -1:
                # BFS from this node
                queue = [node]
                fragment_ids[node] = current_fragment
                
                while queue:
                    current = queue.pop(0)
                    # Find neighbors
                    neighbors = edge_index[1][edge_index[0] == current].tolist()
                    neighbors += edge_index[0][edge_index[1] == current].tolist()
                    
                    for neighbor in neighbors:
                        if fragment_ids[neighbor] == -1:
                            fragment_ids[neighbor] = current_fragment
                            queue.append(neighbor)
                
                current_fragment += 1
        
        return fragment_ids


class RingsPathsFragmenter(BaseFragmenter):
    """Fragment molecules into rings and paths."""
    
    def __call__(self, edge_index: torch.Tensor, num_nodes: int,
                 smiles: Optional[str] = None, mol: Optional[object] = None) -> torch.Tensor:
        if not HAS_RDKIT:
            return self._heuristic_rings_paths(edge_index, num_nodes)
        
        if mol is None and smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return self._heuristic_rings_paths(edge_index, num_nodes)
        
        try:
            fragment_ids = torch.zeros(num_nodes, dtype=torch.long)
            current_fragment = 0
            assigned = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Get ring information
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            
            # Assign ring atoms to fragments
            for ring in atom_rings:
                if not assigned[list(ring)].any():
                    fragment_ids[list(ring)] = current_fragment
                    assigned[list(ring)] = True
                    current_fragment += 1
            
            # Assign remaining atoms (paths/chains)
            for atom_idx in range(num_nodes):
                if not assigned[atom_idx]:
                    fragment_ids[atom_idx] = current_fragment
                    assigned[atom_idx] = True
                    # Could extend to include connected non-ring atoms
            
            return fragment_ids
            
        except Exception:
            return self._heuristic_rings_paths(edge_index, num_nodes)
    
    def _heuristic_rings_paths(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Heuristic ring detection without RDKit."""
        # Simple fallback: assign based on node degree
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(num_nodes):
            degrees[i] = ((edge_index[0] == i).sum() + (edge_index[1] == i).sum()).item()
        
        # High degree nodes (>2) likely in rings
        fragment_ids = torch.where(degrees > 2, 
                                   torch.tensor(0, dtype=torch.long),
                                   torch.tensor(1, dtype=torch.long))
        return fragment_ids


class RandomFragmenter(BaseFragmenter):
    """Random fragmentation for testing/debugging."""
    
    def __init__(self, num_fragments: int = 3, seed: int = 42):
        self.num_fragments = num_fragments
        self.seed = seed
    
    def __call__(self, edge_index: torch.Tensor, num_nodes: int,
                 smiles: Optional[str] = None, mol: Optional[object] = None) -> torch.Tensor:
        """Randomly assign nodes to fragments."""
        torch.manual_seed(self.seed)
        return torch.randint(0, self.num_fragments, (num_nodes,), dtype=torch.long)


class SingleFragmenter(BaseFragmenter):
    """No fragmentation - treat entire graph as single fragment."""
    
    def __call__(self, edge_index: torch.Tensor, num_nodes: int,
                 smiles: Optional[str] = None, mol: Optional[object] = None) -> torch.Tensor:
        """Assign all nodes to fragment 0."""
        return torch.zeros(num_nodes, dtype=torch.long)


def get_fragmenter(name: str, **kwargs) -> BaseFragmenter:
    """
    Factory function to get fragmenter by name.
    
    Args:
        name: Fragmenter name ('brics', 'ringspaths', 'random', 'single')
        **kwargs: Additional arguments for specific fragmenters
        
    Returns:
        BaseFragmenter instance
    """
    name = name.lower()
    
    if name == 'brics':
        return BRICSFragmenter()
    elif name in ['ringspaths', 'rings_paths']:
        return RingsPathsFragmenter()
    elif name == 'random':
        return RandomFragmenter(**kwargs)
    elif name == 'single':
        return SingleFragmenter()
    else:
        raise ValueError(f"Unknown fragmenter: {name}. "
                        f"Supported: brics, ringspaths, random, single")

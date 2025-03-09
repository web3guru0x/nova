# interpretation.py - Updated for GPU processing

import torch
import pickle
from rdkit import Chem
import numpy as np

def read_molecule(filepath):
    """
    Read a serialized molecule from a pickle file with GPU-optimized processing
    
    Args:
        filepath (str): Path to the pickle file
        
    Returns:
        tuple: (molecule, atom_scores) tuple containing the RDKit molecule and scores
    """
    # Pickle loading is CPU-only
    with open(filepath, 'rb') as f:
        molecule = pickle.load(f)
    
    # Extract atom properties and move to GPU for processing if needed
    atom_props = {}
    atom_scores = []
    
    for atom in molecule.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_props[atom.GetIdx()] = props
        
        if 'PSICHIC_Atom_Score' in props:
            # Store as tensor for efficient GPU processing
            atom_scores.append(float(props['PSICHIC_Atom_Score']))
    
    if atom_scores:
        # Convert to tensor for further GPU processing if needed
        atom_scores = torch.tensor(atom_scores, device='cuda:0')
    
    return molecule, atom_scores

def load_fingerprint(filepath):
    """
    Load a fingerprint from a numpy file and convert to GPU tensor
    
    Args:
        filepath (str): Path to the .npy file
        
    Returns:
        torch.Tensor: Fingerprint tensor on GPU
    """
    # NumPy loading is CPU-only
    fingerprint = np.load(filepath)
    
    # Move to GPU for further processing
    return torch.from_numpy(fingerprint).to('cuda:0')

def compare_molecules(fp1, fp2):
    """
    Compare two fingerprints using GPU-accelerated similarity metrics
    
    Args:
        fp1 (torch.Tensor): First fingerprint tensor
        fp2 (torch.Tensor): Second fingerprint tensor
        
    Returns:
        float: Similarity score
    """
    # Ensure tensors are on GPU
    if not fp1.is_cuda:
        fp1 = fp1.to('cuda:0')
    if not fp2.is_cuda:
        fp2 = fp2.to('cuda:0')
    
    # Compute cosine similarity (very efficient on GPU)
    similarity = torch.nn.functional.cosine_similarity(fp1, fp2, dim=0)
    
    return similarity.item()
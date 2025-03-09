import torch

EPS = 1e-15
from .layers import *

def dense_mincut_pool(x, adj, s, mask=None, cluster_drop_node=None):
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper, optimized for GPU processing.

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        cluster_drop_node (BoolTensor, optional): Optional mask for cluster nodes.

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`, :class:`Tensor`)
    """
    # Use CUDA stream for parallel processing
    with torch.cuda.stream(torch.cuda.Stream()):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        # Apply softmax for numerical stability
        s = torch.softmax(s, dim=-1)

        if mask is not None:
            mask_view = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask_view
            
            x_mask = mask_view
            if cluster_drop_node is not None:
                x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)
            
            x = x * x_mask

        # Matrix multiplications are very efficient on GPU
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # MinCut regularization
        mincut_num = _rank3_trace(out_adj)
        d_flat = torch.einsum('ijk->ij', adj)
        d = _rank3_diag(d_flat)
        mincut_den = _rank3_trace(
            torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
        mincut_loss = -(mincut_num / mincut_den)
        mincut_loss = torch.mean(mincut_loss)

        # Orthogonality regularization
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k, device=ss.device)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # Fix and normalize coarsened adjacency matrix
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d).unsqueeze(2) + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, mincut_loss, ortho_loss

def _rank3_trace(x):
    return torch.einsum('ijj->i', x)

def _rank3_diag(x):
    eye = torch.eye(x.size(1), device=x.device)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out

def dense_dmon_pool(x, adj, s, mask=None):
    r"""
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in
            \mathbb{R}^{B \times N \times F}` with batch-size
            :math:`B`, (maximum) number of nodes :math:`N` for each graph,
            and feature dimension :math:`F`.
        adj (Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
    """
    # Use CUDA stream and automatic mixed precision
    with torch.cuda.stream(torch.cuda.Stream()):
        with torch.cuda.amp.autocast():
            x = x.unsqueeze(0) if x.dim() == 2 else x
            adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
            s = s.unsqueeze(0) if s.dim() == 2 else s

            (batch_size, num_nodes, _), k = x.size(), s.size(-1)
            s = torch.softmax(s, dim=-1)
            s_out = s 
            
            if mask is not None:
                mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
                x, s = x * mask, s * mask

            out = torch.matmul(s.transpose(1, 2), x)
            out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

            # Spectral loss:
            degrees = torch.einsum('ijk->ik', adj).transpose(0, 1)
            m = torch.einsum('ij->', degrees)

            ca = torch.matmul(s.transpose(1, 2), degrees)
            cb = torch.matmul(degrees.transpose(0, 1), s)

            normalizer = torch.matmul(ca, cb) / 2 / m
            decompose = out_adj - normalizer
            spectral_loss = -_rank3_trace(decompose) / 2 / m
            spectral_loss = torch.mean(spectral_loss)

            # Orthogonality regularization:
            ss = torch.matmul(s.transpose(1, 2), s)
            i_s = torch.eye(k, device=ss.device)
            ortho_loss = torch.norm(
                ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
                i_s / torch.norm(i_s), dim=(-1, -2))
            ortho_loss = torch.mean(ortho_loss)

            # Cluster loss:
            cluster_loss = torch.norm(torch.einsum(
                'ijk->ij', ss)) / adj.size(1) * torch.norm(i_s) - 1

            # Fix and normalize coarsened adjacency matrix:
            ind = torch.arange(k, device=out_adj.device)
            out_adj[:, ind, ind] = 0
            d = torch.einsum('ijk->ij', out_adj)
            d = torch.sqrt(d).unsqueeze(2) + EPS
            out_adj = (out_adj / d) / d.transpose(1, 2)

    return s_out, out, out_adj, spectral_loss, ortho_loss, cluster_loss

def simplify_pool(x, adj, s, mask=None, normalize=True):
    r"""The Just Balance pooling operator
    
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """
    # Use CUDA stream for efficient computation
    with torch.cuda.stream(torch.cuda.Stream()):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        s = torch.softmax(s, dim=-1)
        s_out = s 

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        # Use matrix multiplication for efficient GPU computation
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        
        # Loss computation
        ss = torch.matmul(s.transpose(1, 2), s)
        # Use safe sqrt with EPS to avoid numerical issues
        ss_sqrt = torch.sqrt(ss + EPS)
        loss = torch.mean(-_rank3_trace(ss_sqrt))
        if normalize:
            loss = loss / torch.sqrt(torch.tensor(num_nodes * k, device=x.device))

        # Fix and normalize coarsened adjacency matrix
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d).unsqueeze(2) + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

    return s_out, out, out_adj, loss
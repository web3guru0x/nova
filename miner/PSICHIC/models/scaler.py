from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.utils import degree


class DegreeScalerAggregation(Aggregation):
    r"""Combines one or more aggregators and transforms its output with one or
    more scalers as introduced in the `"Principal Neighbourhood Aggregation for
    Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.
    Optimized for GPU processing.

    Args:
        aggr (string or list or Aggregation): The aggregation scheme to use.
        scaler (str or list): Set of scaling function identifiers.
        deg (Tensor): Histogram of in-degrees of nodes in the training set.
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function. (default: :obj:`None`)
    """
    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        # Optimize aggregator initialization for GPU
        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(f"Only strings, list, tuples and instances of"
                             f"`torch_geometric.nn.aggr.Aggregation` are "
                             f"valid aggregation schemes (got '{type(aggr)}')")

        self.scaler = [scaler] if isinstance(scaler, str) else scaler

        # Pre-compute average degrees for GPU
        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel(), device=deg.device)
        self.avg_deg = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }
        
        # Cache parameters in registered buffers for GPU access
        self.register_buffer('bin_degrees', bin_degrees)
        self.register_buffer('deg_values', deg)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        # Use CUDA streams for parallel processing
        with torch.cuda.stream(torch.cuda.Stream()):
            # Assert index is present for degree computation
            self.assert_index_present(index)

            # Use the aggregator
            out = self.aggr(x, index, ptr, dim_size, dim)

            # Compute degrees with careful device management
            assert index is not None
            deg = degree(index, num_nodes=dim_size, dtype=out.dtype).clamp_(1)
            size = [1] * len(out.size())
            size[dim] = -1
            deg = deg.view(size)

            # Process scalers in parallel when possible
            outs = []
            for scaler in self.scaler:
                if scaler == 'identity':
                    out_scaler = out
                elif scaler == 'amplification':
                    log_deg = torch.log(deg + 1)
                    out_scaler = out * (log_deg / self.avg_deg['log'])
                elif scaler == 'attenuation':
                    log_deg = torch.log(deg + 1)
                    out_scaler = out * (self.avg_deg['log'] / log_deg)
                elif scaler == 'exponential':
                    exp_deg = torch.exp(deg)
                    out_scaler = out * (exp_deg / self.avg_deg['exp'])
                elif scaler == 'linear':
                    out_scaler = out * (deg / self.avg_deg['lin'])
                elif scaler == 'inverse_linear':
                    out_scaler = out * (self.avg_deg['lin'] / deg)
                else:
                    raise ValueError(f"Unknown scaler '{scaler}'")
                outs.append(out_scaler)

        # Combine outputs efficiently
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
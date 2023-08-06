import numba
import numpy as np
import scipy.sparse as sp
from collections import Counter
import torch
# from tqdm import tqdm
# from torch_scatter import scatter_add
import dgl

@numba.njit(cache=True,
            locals={
                '_val': numba.float32,
                'res': numba.float32,
                'res_vnode': numba.float32
            })
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
  alpha_eps = alpha * epsilon
  f32_0 = numba.float32(0)
  p = {inode: f32_0}
  r = {}
  r[inode] = alpha
  q = [inode]
  while len(q) > 0:
    unode = q.pop()

    res = r[unode] if unode in r else f32_0
    if unode in p:
      p[unode] += res
    else:
      p[unode] = res
    r[unode] = f32_0
    for vnode in indices[indptr[unode]:indptr[unode + 1]]:
      _val = (1 - alpha) * res / deg[unode]
      if vnode in r:
        r[vnode] += _val
      else:
        r[vnode] = _val

      res_vnode = r[vnode] if vnode in r else f32_0
      if res_vnode >= alpha_eps * deg[vnode]:
        if vnode not in q:
          q.append(vnode)

  return list(p.keys()), list(p.values())


@numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
  js = []
  vals = []
  for i, node in enumerate(nodes):
    j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
    js.append(j)
    vals.append(val)
  return js, vals


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
  js = [np.zeros(0, dtype=np.int64)] * len(nodes)
  vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
  for i in numba.prange(len(nodes)):
    j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
    j_np, val_np = np.array(j), np.array(val)
    idx_topk = np.argsort(val_np)[-topk:]
    js[i] = j_np[idx_topk]
    vals[i] = val_np[idx_topk]
  return js, vals


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel_keep(indptr, indices, deg, alpha, epsilon, nodes,
                                keep_nodes, topk):
  """Keep only certain nodes"""
  js = [np.zeros(0, dtype=np.int64)] * len(nodes)
  vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
  for i in numba.prange(len(nodes)):
    j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
    for k in range(len(j)):
      if j[k] in keep_nodes:
        pass
      else:
        val[k] = 0
    j_np, val_np = np.array(j), np.array(val)
    idx_topk = np.argsort(val_np)[-topk:]
    js[i] = j_np[idx_topk]
    vals[i] = val_np[idx_topk]
  return js, vals


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk, keep_nodes=None):
  """Calculate the PPR matrix approximately using Anderson."""

  out_degree = np.sum(adj_matrix > 0, axis=1).A1
  nnodes = adj_matrix.shape[0]

  if keep_nodes:
    keep_nodes = set(keep_nodes)
    neighbors, weights = calc_ppr_topk_parallel_keep(adj_matrix.indptr,
                                                     adj_matrix.indices,
                                                     out_degree,
                                                     numba.float32(alpha),
                                                     numba.float32(epsilon),
                                                     nodes, keep_nodes, topk)
  else:
    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr,
                                                adj_matrix.indices, out_degree,
                                                numba.float32(alpha),
                                                numba.float32(epsilon), nodes,
                                                topk)

  return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
  i = np.repeat(np.arange(len(neighbors)),
                np.fromiter(map(len, neighbors), dtype=np.int))
  j = np.concatenate(neighbors)
  return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


# def topk_ppr_matrix(edge_index, path_len, num_node,device):
#   adj_dict = {}
#
#   def add_edge(a, b):
#     if a in adj_dict:
#       neighbors = adj_dict[a]
#     else:
#       neighbors = set()
#       adj_dict[a] = neighbors
#     if b not in neighbors:
#       neighbors.add(b)
#
#
#   for a, b in zip(edge_index[0].numpy(), edge_index[1].numpy()):
#     a = int(a)
#     b = int(b)
#     add_edge(a, b)
#     add_edge(b, a)
#   adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}
#
#   def sample_neighbor(a):
#     neighbors = adj_dict[a]
#     random_index = np.random.randint(0, len(neighbors))
#     return neighbors[random_index]
#
#   walk_counters = {}
#
#   def norm(counter):
#     s = sum(counter.values())
#     new_counter = Counter()
#     for a, count in counter.items():
#       new_counter[a] = counter[a] / s
#     return new_counter
#
#   for _ in tqdm(range(40)):
#     for a in adj_dict:
#       current_a = a
#       current_path_len = np.random.randint(1, path_len + 1)
#       for _ in range(current_path_len):
#         b = sample_neighbor(current_a)
#         if a in walk_counters:
#           walk_counter = walk_counters[a]
#         else:
#           walk_counter = Counter()
#           walk_counters[a] = walk_counter
#         walk_counter[b] += 1
#         current_a = b
#
#   normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}
#
#   prob_sums = Counter()
#
#   for a, normed_walk_counter in normed_walk_counters.items():
#     for b, prob in normed_walk_counter.items():
#       prob_sums[b] += prob
#
#   ppmis = {}
#
#   for a, normed_walk_counter in normed_walk_counters.items():
#     for b, prob in normed_walk_counter.items():
#       ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / path_len)
#       ppmis[(a, b)] = ppmi
#
#   new_edge_index = []
#   edge_weight = []
#   for (a, b), ppmi in ppmis.items():
#     new_edge_index.append([a, b])
#     edge_weight.append(ppmi)
#
#   new_edge_index = torch.tensor(new_edge_index).t()
#   edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
#
#   row, col = new_edge_index
#   deg = scatter_add(edge_weight, row, dim=0)
#   deg_inv_sqrt = deg.pow(-0.5)
#   deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#
#   edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#   weight_index = torch.where(edge_weight > 0.)
#   edge_weight = edge_weight[weight_index]
#
#   row_list=[]
#   col_list=[]
#   for i in range(num_node):
#     row_index=torch.where(row[weight_index]==i)
#     weight=edge_weight[row_index[0]]
#     if weight.shape[0]>5:
#       topk=row_index[0][torch.topk(weight,5).indices]
#     else:
#       topk=row_index[0]
#     col_index=col[weight_index][topk]
#     row_list+=torch.tensor(i).repeat(topk.shape).tolist()
#     col_list += col_index.tolist()
#   data = dgl.graph((row_list,col_list),num_nodes=num_node)
#   data=dgl.to_bidirected(data)
#   return data




# def topk_ppr_matrix(adj_matrix,
#                     alpha,
#                     eps,
#                     idx,
#                     topk,
#                     normalization='row',
#                     keep_nodes=None):
#   """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
#
#   topk_matrix = ppr_topk(adj_matrix,
#                          alpha,
#                          eps,
#                          idx,
#                          topk,
#                          keep_nodes=keep_nodes).tocsr()
#
#   if normalization == 'sym':
#     # Assume undirected (symmetric) adjacency matrix
#     deg = adj_matrix.sum(1).A1
#     deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
#     deg_inv_sqrt = 1. / deg_sqrt
#
#     row, col = topk_matrix.nonzero()
#     # assert np.all(deg[idx[row]] > 0)
#     # assert np.all(deg[col] > 0)
#     topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
#   elif normalization == 'col':
#     # Assume undirected (symmetric) adjacency matrix
#     deg = adj_matrix.sum(1).A1
#     deg_inv = 1. / np.maximum(deg, 1e-12)
#
#     row, col = topk_matrix.nonzero()
#     # assert np.all(deg[idx[row]] > 0)
#     # assert np.all(deg[col] > 0)
#     topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
#   elif normalization == 'row':
#     pass
#   else:
#     raise ValueError(f"Unknown PPR normalization: {normalization}")
#
#   return topk_matrix

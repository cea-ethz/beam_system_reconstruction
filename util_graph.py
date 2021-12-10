"""
Utilities for working with NetworkX graphs
"""

import networkx as nx
import numpy as np


def get_stream_counts(dg, nid):
    """
    Return total number of upstream and downstream nodes
    :param dg: Direction graph to query
    :param nid: Node id within graph to start from
    :return:
    """
    upstream = [n for n in nx.traversal.bfs_tree(dg, nid, reverse=True) if n != nid]
    downstream = [n for n in nx.traversal.bfs_tree(dg, nid) if n != nid]
    return len(upstream), len(downstream)


def get_node_id(dg, outer_id):
    return list(dg.nodes).index(outer_id)


def get_edge_id(dg, a, b):
    for i, edge in enumerate(dg.edges):
        if edge[0] == a and edge[1] == b:
            return i
    return None


def simplify_position(dg, pos, node_ids):

    for nid in node_ids:
        edges = dg.out_edges(nid)
        y = [pos[n[1]][1] for n in edges]
        if len(y):
            pos[nid][1] = sum(y) / len(y)
        else:
            pos[nid][1] = - 10
    return pos


def normalize_position(dg, pos, node_ids):
    y = np.array([pos[nid][1] for nid in node_ids])
    args = np.argsort(y)

    if len(node_ids) - 1 == 0:
        return pos

    for i, arg_id in enumerate(args):
        n = 1.0 * i / (len(node_ids) - 1)
        pos[node_ids[arg_id]][1] = n * 2 - 1

    return pos

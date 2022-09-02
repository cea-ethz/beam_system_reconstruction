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


def simplify_position(dg, pos, node_ids, vertical):
    dir = 1 if vertical else 0

    for nid in node_ids:
        edges = dg.in_edges(nid)
        coord = [pos[n[0]][dir] for n in edges]
        if len(coord):
            pos[nid][dir] = sum(coord) / len(coord)
        else:
            pos[nid][dir] = - 10
    return pos


def normalize_position(dg, pos, node_ids, vertical):
    dir = 1 if vertical else 0
    coord = np.array([pos[nid][dir] for nid in node_ids])
    args = np.argsort(coord)

    if len(node_ids) - 1 == 0:
        return pos

    for i, arg_id in enumerate(args):
        n = 1.0 * i / (len(node_ids) - 1)
        pos[node_ids[arg_id]][dir] = n * 2 - 1

    return pos

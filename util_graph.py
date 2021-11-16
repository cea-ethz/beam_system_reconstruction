"""
Utilities for working with NetworkX graphs
"""

import networkx as nx


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


def get_node_id(dg,outer_id):
    return list(dg.nodes).index(outer_id)


def get_edge_id(dg, a, b):
    for i, edge in enumerate(dg.edges):
        if edge[0] == a and edge[1] == b:
            return i
    return None

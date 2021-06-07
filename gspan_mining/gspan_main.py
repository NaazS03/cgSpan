"""The main program that runs gSpan."""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from config import parser
from gspan import gSpan


def main(FLAGS=None):
    """Run gSpan."""

    if FLAGS is None:
        FLAGS, _ = parser.parse_known_args(args=sys.argv[1:])

    if not os.path.exists(FLAGS.database_file_name):
        print('{} does not exist.'.format(FLAGS.database_file_name))
        sys.exit()

    gs = gSpan(
        database_file_name=FLAGS.database_file_name,
        min_support=FLAGS.min_support,
        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
        max_num_vertices=FLAGS.upper_bound_of_num_vertices,
        max_ngraphs=FLAGS.num_graphs,
        is_undirected=(not FLAGS.directed),
        verbose=FLAGS.verbose,
        visualize=FLAGS.plot,
        where=FLAGS.where
    )

    gs.run()
    gs.time_stats()
    #not_closed_frequent_graphs = find_not_closed_frequent_graphs(gs)
    not_closed_frequent_graphs = find_not_closed_frequent_graphs_support_graphs(gs)
    num_closed_graphs = len(gs._frequent_subgraphs) - len(not_closed_frequent_graphs)
    print("number of closed graphs ", num_closed_graphs)
    closed_graphs = list()
    for g in gs._frequent_subgraphs:
        if not g.gid in not_closed_frequent_graphs:
            closed_graphs.append(g)
    closed_graphs.sort(key=lambda x: (x.support_projections, x.get_num_vertices()))
    for g in closed_graphs:
        g.display()
        print("projections ", g.support_projections, " vertices ", g.get_num_vertices())
        print("graphs ", g.support_graphs, " vertices ", g.get_num_vertices())
        print("------------------")
    return gs

def find_not_closed_frequent_graphs(gs):
    min_subgraph_size, max_subgraph_size = find_min_and_max_size_subgraph(gs)
    graphs_by_size = dict()
    for i in range(min_subgraph_size, max_subgraph_size + 1):
        graphs_by_size[i] = list()
    for subgraph in gs._frequent_subgraphs:
        graphs_by_size[subgraph.num_edges].append(subgraph)
    not_closed_graphs = set()
    for i in range(min_subgraph_size, max_subgraph_size + 1):
        if i < max_subgraph_size:
            find_not_closed_graphs(graphs_by_size[i + 1], graphs_by_size[i], not_closed_graphs)
    return not_closed_graphs

def find_not_closed_graphs(graphs_big, graphs_small, not_closed_graphs):
    for graph_big in graphs_big:
        for graph_small in graphs_small:
            if graph_small.gid in not_closed_graphs:
                continue
            if graph_small.gid == graph_big.gid:
                continue
            if graph_big.is_supergraph_of_with_support_projections(graph_small):
                not_closed_graphs.add(graph_small.gid)



def find_not_closed_frequent_graphs_support_graphs(gs):
    min_subgraph_size, max_subgraph_size = find_min_and_max_size_subgraph(gs)
    graphs_by_size = dict()
    for i in range(min_subgraph_size, max_subgraph_size + 1):
        graphs_by_size[i] = list()
    for subgraph in gs._frequent_subgraphs:
        graphs_by_size[subgraph.num_edges].append(subgraph)
    not_closed_graphs = set()
    for i in range(min_subgraph_size, max_subgraph_size + 1):
        if i < max_subgraph_size:
            find_not_closed_graphs_support_graphs(graphs_by_size[i + 1], graphs_by_size[i], not_closed_graphs)
    return not_closed_graphs

def find_not_closed_graphs_support_graphs(graphs_big, graphs_small, not_closed_graphs):
    for graph_big in graphs_big:
        for graph_small in graphs_small:
            if graph_small.gid in not_closed_graphs:
                continue
            if graph_small.gid == graph_big.gid:
                continue
            if graph_big.is_supergraph_of_with_support_projections(graph_small):
                not_closed_graphs.add(graph_small.gid)



def find_min_and_max_size_subgraph(gs):
    min_subgraph_size = gs._frequent_subgraphs[0].num_edges
    max_subgraph_size = gs._frequent_subgraphs[0].num_edges

    for subgraph in gs._frequent_subgraphs:
        subgraph_size = subgraph.num_edges

        if subgraph_size > max_subgraph_size:
            max_subgraph_size = subgraph_size

        elif subgraph_size < min_subgraph_size:
            min_subgraph_size = subgraph_size

    return min_subgraph_size, max_subgraph_size

if __name__ == '__main__':
    main()

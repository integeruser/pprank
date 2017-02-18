#!/usr/bin/env python3
import argparse

import networkx as nx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='the file containing the directed edges')
    parser.add_argument('tol', type=float, default=1e-6, nargs='?', help='the tolerance')
    args = parser.parse_args()

    print('[*] Building graph...')
    graph = nx.read_edgelist(args.file, create_using=nx.DiGraph(), nodetype=int)
    print('        Nodes:    %d' % len(graph.nodes()))
    print('        Edges:    %d' % len(graph.edges()))
    print('        Dangling: %d' % sum(out_degree == 0
                                       for node, out_degree in graph.out_degree().items()))

    print("[*] Computing PageRanks (tol=%g)..." % args.tol)
    ranks = nx.pagerank(graph, tol=args.tol)

    print('[*] PageRanks:')
    for node, rank in sorted(ranks.items()):
        print('        %0.9d: %g' % (node, rank))

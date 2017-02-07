#!/usr/bin/env python3
import argparse
import collections
import os
import re

edge = re.compile('(\d+)\s+(\d+)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='the data set to sort')
    args = parser.parse_args()

    edges = collections.defaultdict(set)
    with open(args.file) as f:
        for line in f:
            match = edge.match(line)
            if match:
                from_node, to_node = [int(node) for node in match.groups()]
                edges[from_node].add(to_node)

    with open(f'ordered-{os.path.basename(args.file)}', 'w') as f:
        for from_node in sorted(edges):
            for to_node in sorted(edges[from_node]):
                f.write(f'{from_node} {to_node}\n')

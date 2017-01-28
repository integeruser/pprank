#!/usr/bin/env python3
import argparse
import os
import re

edge = re.compile('(\d+)\s+(\d+)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='the one-based data set to convert from to zero-based')
    args = parser.parse_args()

    with open(args.file) as f1, open(f'zero-based-{os.path.basename(args.file)}', 'w') as f2:
        for line in f1:
            match = edge.match(line)
            if match:
                id1, id2 = match.groups()
                new_id1, new_id2 = int(id1) - 1, int(id2) - 1
                f2.write(f'{new_id1} {new_id2}\n')
            else:
                f2.write(line)

"""
Short python script that merges multiple csvs of classifications. Not really
used but created more as a historical record of how datasets are combined when
labels are sent in a piecemeal fashion.

First checks all given label_dirs that they both contain a common set of files,
then it reads all the files, assuming they are csv files, it concatenates them
all together and writes the new concatenated csv files to out_dir.
"""
import os
import pandas as pd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dirs', type=str, nargs='+', required=True,
                        help="""
                        Paths to directories that contain all output csv label
                        files. Probably all csv files were made from the
                        json2csv.py script.
                        """)
    parser.add_argument('--out_dir', type=str, required=True,
                        help="""
                        Path to directory where the joined csvs should be
                        written.
                        """)
    args = parser.parse_args()

    for i, dir in enumerate(args.label_dirs):
        if i == 0:
            bn_set = set(os.listdir(dir))
        else:
            new_bn_set = set(os.listdir(dir))
            assert len(bn_set.difference(new_bn_set)) == 0, (
                    f'Found missing files in {dir}')
            assert len(new_bn_set.difference(bn_set)) == 0, (
                    f'Found too many files in {dir}')
    os.makedirs(args.out_dir)
    for bn in bn_set:
        dfs = [pd.read_csv(os.path.join(d, bn)) for d in args.label_dirs]
        df = pd.concat(dfs)
        df.to_csv(os.path.join(args.out_dir, bn))




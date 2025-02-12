
from docopt import docopt
import os
import sys


DOCTEXT = """
    Usage:
      create_dataset_from_folders.py <input> <output> --color=<c>... [--by_id | --only_by_id] [--label]
"""


def print_dir(dir_path, color, file=None, label=False):
    with os.scandir(dir_path) as dir_:
        for entry in dir_:
            if entry.is_file():
                subdir = os.path.basename(os.path.dirname(dir_path))
                line = f"{subdir}/{entry.name},{color}" if label else entry.name
                print(line, end='\n', file=file)

def main(input, output, colors, by_id=False, only_by_id=False, label=False):
    with open(output, 'w') as f:
        for color in colors:
            if not only_by_id:
                print_dir(f"{input}/{color}/", color, file=f, label=label)
            
            if by_id:
                try:
                    print_dir(f"{input}/{color}_by_id/", color, file=f, label=label)
                except FileNotFoundError:
                    pass

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    input = args["<input>"]
    output = args["<output>"]
    colors = args["--color"] if args["--color"] != 0 else ['inference']
    only_by_id = args["--only_by_id"]
    by_id = args["--by_id"] or only_by_id
    label = args["--label"]

    main(input, output, colors, by_id=by_id, only_by_id=only_by_id, label=label)

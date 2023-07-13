
from collections.abc import MutableMapping
from docopt import docopt
import json
import os
import pandas as pd
from pathlib import Path
import sys
import wandb
import yaml


def flatten(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


DOCTEXT = f"""
Usage:
  wandb_log_fast_reid.py <output_path>

"""

if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    output_path = args['<output_path>']

    output_path = os.path.join(*output_path[1:].strip("'").split('/'))

    yaml_path = os.path.join(output_path, "config.yaml")
    metrics_path = os.path.join(output_path, "metrics.json")

    config = yaml.safe_load(Path(yaml_path).read_text())
    config = flatten(config)
    wandb.init(config=config, resume=True)
    wandb.config['OUTPUT_DIR_FINAL'] = output_path

    with open(metrics_path) as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    df = df.sort_values('iteration', ignore_index=True)
    df = df.groupby('iteration', as_index=False).median()

    for _, row in df.iterrows():
        metrics = row[row.notna()].to_dict()
        iteration = metrics['iteration']

        wandb.log(metrics, step=int(iteration))

    best = list(df[df['mINP'] == df['mINP'].max()].iterrows())[0][1].to_dict()
    for key, val in best.items():
        wandb.run.summary[key] = val

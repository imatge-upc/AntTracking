
from docopt import docopt
import json
import os
import pandas as pd
import sys
import wandb


DOCTEXT = f"""
Usage:
  wandb_log_fast_reid.py <output_path>

"""

if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    output_path = args['<output_path>']

    yaml_path = os.path.join(output_path, "config.yaml")
    metrics_path = os.path.join(output_path, "metrics.json")

    wandb.init(config=yaml_path)

    with open('./runs/apparence/train01_colonia_256_128/metrics.json') as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    df = df.sort_values('iteration', ignore_index=True)
    df = df.groupby('iteration', as_index=False).median()

    for _, row in df.iterrows():
        metrics = row[row.notna()].to_dict()
        iteration = metrics['iteration']

        wandb.log(metrics, step=iteration)

    best = list(df[df['mINP'] == df['mINP'].max()].iterrows())[0][1].to_dict()
    for key, val in best.items():
        wandb.run.summary[key] = val

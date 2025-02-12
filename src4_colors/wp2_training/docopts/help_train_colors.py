
from docopt import docopt


DOCTEXT = """
    Usage:
      train_colors.py <dataset_path> <data_path> <basename> [options]
      train_colors.py -h | --help

    Options:
      -n <n>, --num_gauss=<n>       Number of gaussians for each GMM [default: 4]
      --cov_type=<c>                {full, tied, diag, spherical}, [default: full]
      --num_iters=<i>               The number of EM iterations to perform [default: 1000]
      --num_init=<ni>               The number of initializations to perform. The best results are kept. [default: 10]
      --verbose=<v>                 From 1 to 31 enable some verbose output. Each bit enable different levels {LAST_VERBOSE, SAVE_VERBOSE, MEAN_VERBOSE, SK2_VERBOSE, SK1_VERBOSE}. SK2_VERBOSE overwrites SK1_VERBOSE. [default: 0]
      --verbose_interval=<vi>       Number of iteration done before the next print for SK1_VERBOSE and SK2_VERBOSE. [default: 200]

"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    dataset_path = args["<dataset_path>"]
    data_path = args["<data_path>"]
    basename = args["<basename>"]
    num_gauss = int(args["--num_gauss"])
    cov_type = args["--cov_type"]
    num_iters = int(args["--num_iters"])
    num_init = int(args["--num_init"])
    verbose = int(args["--verbose"])
    sk_verbose_interval = int(args["--verbose_interval"])

    return (dataset_path, data_path, basename, num_gauss, cov_type, num_iters, num_init, verbose, sk_verbose_interval)

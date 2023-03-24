import argparse

import utils

from config import cfg

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=add_help)

    parser.add_argument('--config-file', help="""Configuration file name in config/ folder.""", type=str)
    return parser

def main(cfg):
    if cfg.OUTPUT_DIR:
        utils.mkdir(cfg.OUTPUT_DIR)
    
    utils.init_random_seed()

    print("Loading dataset...")

    pass

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.config_file:
        cfg = cfg.merge_from_file(args.config_file)

    main(cfg)

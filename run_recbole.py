import argparse

from recbole.quick_start import run_recbole
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRecR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--log_prefix', type=str, default='')

    args, _ = parser.parse_known_args()
    config_dict = {}
    config_file_list = None
    if args.config_files:
        config_file_list = args.config_files.strip().split(' ')
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,
                config_dict=config_dict, log_prefix=args.log_prefix)

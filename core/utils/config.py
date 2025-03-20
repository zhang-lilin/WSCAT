import argparse

import yaml
from .args import _arg_values, _global_parser
from .parser import set_parser_train, set_parser_eval
'''
Argument Definition:
config_file : file to config argument
'''
args = _arg_values()
args.DEFINE_argument("config_file", type=str, help="Path to config file.",)

def load_config(train=True):
    if train:
        set_parser_train(_global_parser)
    else:
        set_parser_eval(_global_parser)

    config_path = args.config_file
    with open(config_path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)
    input_arguments = []
    all_keys = args.get_dict()
    print(all_keys)

    for k in cfg_special:
        if k in all_keys and cfg_special[k] == all_keys[k]:
            continue
        else:
            if k in all_keys:
                args.__setattr__(k, cfg_special[k])
                print("Note!: set args: {} with value {}".format(k, args.__getattr__(k)))
                input_arguments.append(k)
            else:
                v = cfg_special[k]
                if type(v) == bool:
                    args.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
                else:
                    args.DEFINE_argument(
                        "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
                    )
                args.__setattr__(k, cfg_special[k])
                print("Note!: new args: {} with value {}".format(k, args.__getattr__(k)))
    if train:
        check(args)
    return input_arguments

import re
SEEDS = [1,2,3]
def check(args):
    if args.pre_resume_path:
        seed_info = f'seed{args.seed}'
        for s in SEEDS:
            if s != args.seed:
                s_info = f'seed{s}'
                args.pre_resume_path = re.sub(s_info, seed_info, args.pre_resume_path)
        assert seed_info in args.pre_resume_path
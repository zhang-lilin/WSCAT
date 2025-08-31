import argparse
import json
import os
import pickle
import socket
import torch
from core.data import get_data_info
from core.data import load_test_data
from core.metrics import _METRICS
from core.models import create_model
from core.trainer import Logger, seed
from core.trainer.config import args, load_config

# Setup
load_config(train=False)
attack_name = args.attack_name
if not os.path.exists(args.desc):
    print('File not found.')
    exit()

LOG_DIR = os.path.join(args.desc, args.log_dir, )
if not os.path.isdir(LOG_DIR):
    os.mkdir(os.path.join(LOG_DIR))

test_log_path = os.path.join(LOG_DIR, 'log-test-{}.log'.format(attack_name))
stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
weights = {'state-last.pt': 'last', 'weights-best.pt': 'best'}
test_save_paths = [os.path.join(LOG_DIR, f'{attack_name}_{v}.pt') for v in weights.values()]

if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    if attack_name in stats and args.save and any(os.path.exists(p) for p in test_save_paths):
        print('Already tested.')
        exit()

for path in [test_log_path] + test_save_paths:
    if os.path.exists(path):
        os.remove(path)
logger = Logger(test_log_path)

host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
logger.log("#" * 120)
logger.log("----------Configurable Parameters In this Model----------")
logger.log(host_info)
for k in args.get_dict():
    logger.log("# " + ("%30s" % k) + ":\t" + str(args.__getattr__(k)))
logger.log("#" * 120)

ARG_PATH = os.path.join(args.desc, 'args.txt')
with open(ARG_PATH, 'r') as f:
    cfg_special = json.load(f)

model_train_seed = cfg_special['seed']
logger.log(f'Model training seed {model_train_seed}.')

all_keys = args.get_dict()
for k, v in cfg_special.items():
    if k not in all_keys:
        if isinstance(v, bool):
            args.DEFINE_boolean(f"-{k}", f"--{k}", default=argparse.SUPPRESS)
        else:
            args.DEFINE_argument(f"-{k}", f"--{k}", default=argparse.SUPPRESS, type=type(v))
        args.__setattr__(k, v)
        print(f"OLD ARG: {k} with value {v}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log('Using device: {}'.format(device))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
torch.backends.cudnn.benchmark = True

DATA_DIR = os.path.join(args.data_dir, args.data)
info = get_data_info(DATA_DIR)
test_dataset, test_dataloader = load_test_data(DATA_DIR, batch_size_test=args.batch_size_validation, num_workers=0)
model = create_model(args, info, device, None)

for w in weights:
    WEIGHTS = os.path.join(args.desc, w)
    checkpoint = torch.load(WEIGHTS, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint['epoch']
    else:
        raise FileExistsError

    logger.log(f'Resuming target model at {WEIGHTS} (epoch-{epoch}).')
    if 'last' in w and epoch < args.num_epochs:
        os.remove(LOG_DIR)
        print('Training unfinished.')
        exit()

    num_classes = info['num_classes']
    logger.log(f'Begin evaluation: {weights[w]}-{attack_name}.')
    attack_opt = args.attack_opt
    if attack_name == 'aa':
        attack_opt['n_classes'] = num_classes
    adversary = _METRICS[attack_name](model, **attack_opt)

    seed(args.seed)
    model.eval()
    acc, _, _ = adversary.save(data_loader=test_dataloader, verbose=True, return_verbose=True,
                               save_path=test_save_paths[weights[w]] if args.save else None)

    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            logger.stats = pickle.load(f)
    logger.add(category=weights[w], k=attack_name, v=acc, global_it=model_train_seed, unique=True)
    logger.log('Adversarial {}-{}: {:.5f}%'.format(weights[w], attack_name, acc))
    logger.save_stats('eval_stats.pkl')
    logger.log('\nTesting completed.')
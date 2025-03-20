import json
import time
import pickle
import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import argparse
from core.utils.context import ctx_noparamgrad_and_eval
import socket
import torch.nn as nn
from core.models import create_model
from core.data import get_data_info
from core.data import load_test_data
from core.data import DATASETS
from core.utils import format_time
from core.utils import Logger
from core.utils import seed
from core.metrics import accuracy
from core.utils.config import args, load_config

# Setup
args.DEFINE_argument('-device', '--device', type=str, default='none')
load_config(train=False)
# assert args.data in DATASETS, f'Only data in {DATASETS} is supported!'

LOG_DIR = os.path.join(args.log_dir, args.desc)

attack_name, cat = "pgd", 'RAE'
test_log_path = os.path.join(LOG_DIR, 'log-test-{}.log'.format(attack_name))
test_save_dir = os.path.join(LOG_DIR, 'AEs')
if not os.path.isdir(test_save_dir) and args.save:
    os.mkdir(test_save_dir)
test_save_path = os.path.join(test_save_dir, f'{attack_name}.pt')
stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')

if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    if cat in stats:
        if attack_name in stats[cat]:
            if args.save:
                if os.path.exists(test_save_path):
                    print('Already tested.')
                    exit(111)
                else:
                    pass
            else:
                print('Already tested.')
                exit(111)

if os.path.exists(test_log_path):
    os.remove(test_log_path)
if os.path.exists(test_save_path):
    os.remove(test_save_path)
logger = Logger(test_log_path)

host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
logger.log("#" * 120)
logger.log("----------Configurable Parameters In this Model----------")
logger.log(host_info)
for k in args.get_dict():
    logger.log("# " + ("%30s" % k) + ":\t" + str(args.__getattr__(k)))
logger.log("#" * 120)

with open(LOG_DIR+'/args.txt', 'r') as f:
    cfg_special = json.load(f)
    model_train_seed = cfg_special['seed']
    logger.log(f'Model training seed {model_train_seed}.')
    all_keys = args.get_dict()
    for k in cfg_special:
        if k in all_keys:
            pass
        else:
            v = cfg_special[k]
            if type(v) == bool:
                args.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
            else:
                args.DEFINE_argument(
                    "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
                )
            args.__setattr__(k, cfg_special[k])
            print("OLD ARG: {} with value {}".format(k, args.__getattr__(k)))

DATA_DIR = os.path.join(args.data_dir, args.data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.device == 'none':
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

logger.log('Using device: {}'.format(device))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

args.device = device
info = get_data_info(DATA_DIR)
torch.backends.cudnn.benchmark = True
BATCH_SIZE_VALIDATION = args.batch_size_validation
test_dataset, test_dataloader = load_test_data(DATA_DIR, batch_size_test=BATCH_SIZE_VALIDATION, num_workers=4)


if 'standard' not in LOG_DIR and 'mt' not in LOG_DIR:
    WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
else:
    WEIGHTS = os.path.join(LOG_DIR, 'state-last.pt')
model = create_model(args, info, device, None)
checkpoint = torch.load(WEIGHTS)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint['epoch']
elif 'netC' in checkpoint:
    model.load_state_dict(checkpoint["netC"])
    epoch = checkpoint['epoch']
else:
    raise FileExistsError

logger.log(f'Resuming target model at {WEIGHTS} (epoch-{epoch}).')
logger.log(f'Begin the evaluation about {attack_name}.')

from torchattacks import PGD
seed(args.seed)
model.eval()
from core.utils import str2float
adversary = PGD(model, eps=str2float(args.attack_eps), alpha=str2float(args.attack_step), steps=int(args.attack_iter))
acc, _, _ = adversary.save(data_loader=test_dataloader, verbose=True, return_verbose=True,
                           save_path=test_save_path if args.save else None)

seed(args.seed)
acc_origrn = 0.0
model.eval()
total = 0
for x, y in tqdm(test_dataloader, desc="Evaluation : ", disable=False):
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = model(x)
        _, predicted = torch.max(out, 1)
    acc_origrn += (predicted == y).sum().item()
    total += x.size(0)
acc_origrn /= total


stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        logger.stats = pickle.load(f)

logger.add(category=cat, k=attack_name, v=acc, global_it=model_train_seed, unique=True)
logger.log('Adversarial {}: {:.5f}%'.format(attack_name, acc))
logger.add(category="nat", k="nat", v=acc_origrn*100, global_it=model_train_seed, unique=True)
logger.log('Original: {:.5f}%'.format(acc_origrn*100))

logger.save_stats('eval_stats.pkl')
logger.log('\nTesting completed.')
import json
import re
import time
import pickle
import os

import numpy as np
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
from core.utils import format_time, str2float
from core.utils import Logger
from core.utils import seed
from core.metrics import accuracy
from core.utils.config import args, load_config
from core.attacks.IFP import InformativeFeaturePackage


# Setup
args.DEFINE_argument('-device', '--device', type=str, default='none')

load_config(train=False)
# assert args.data in DATASETS, f'Only data in {DATASETS} is supported!'

LOG_DIR = os.path.join(args.log_dir, args.desc)

attack_name, cat = "NRF", 'RAE'
# args.save = True
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
test_dataset, test_dataloader = load_test_data(DATA_DIR, batch_size_test=BATCH_SIZE_VALIDATION, num_workers=4, shuffle=True)


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

seed(args.seed)
model.eval()
args.attack_eps = str2float(args.attack_eps)
IFM = InformativeFeaturePackage(model,
                                # eps=args.attack_eps, attack_iter=args.attack_iter,
                                device=device)

def distance(x_adv, x, attack):
    diff = (x_adv - x).view(x.size(0), -1)
    if attack in ('NRF', 'NRF2', 'cw', 'fab'):
        out = torch.sqrt((torch.sum(diff * diff, dim=1)/diff.size(1)).sum()/diff.size(0)).item()
        return out
    elif attack in ('fgsm', 'bim', 'pgd', 'auto'):
        out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
        return out
    else:
        out = torch.sqrt((torch.sum(diff * diff, dim=1)/diff.size(1)).sum()/diff.size(0)).item()
        return out

def attack_loader(attack, net):
    if attack == 'NRF':
        def f_attack(input, target):
            return net.NRF(input, target)
        return f_attack

    elif attack == 'NRF2':
        def f_attack(input, target):
            return net.NRF2(input, target)
        return f_attack

    elif attack == 'RF':
        def f_attack(input, target):
            return net.RF(input, target)
        return f_attack

    elif args.attack == 'RF2':
        def f_attack(input, target):
            return net.RF2(input, target)
        return f_attack

def experiment_robustness(args, model, attack_name, IFM, save_path=None):
    attack_score = []
    # stack attack module
    attack_module = {}
    attack_module[attack_name] = attack_loader(attack_name, IFM)

    # Measuring L2 distance
    l2_distance_list = []
    for key in attack_module:
        if save_path is not None:
            AEs, AE_labels, AE_preds = None, None, None
        l2_distance = 0
        correct = 0
        total = 0
        print('\n[IFD/Test] Under Testing ... Wait PLZ')
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.to(device), targets.to(device)
            adv_x = attack_module[key](inputs, targets) if args.attack_eps != 0 else inputs
            l2_distance += distance(adv_x, inputs, key)

            # Evaluation
            outputs = model(adv_x)
            pred = torch.max(outputs, dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()
            if save_path is not None:
                AEs = adv_x if AEs is None else torch.cat([AEs, adv_x], 0)
                AE_labels = targets if AE_labels is None else torch.cat([AE_labels, targets], 0)
                AE_preds = pred if AE_preds is None else torch.cat([AE_preds, pred], 0)

        print('[IFD/{}] Acc: {:.3f} ({:.3f})'.format(key, 100. * correct / total, l2_distance / (batch_idx + 1)))
        attack_score.append(100. * correct / total)
        l2_distance_list.append(l2_distance / (batch_idx + 1))
        # if save_path is not None:
        #     torch.save({'AEs':AEs, 'labels': AE_labels, 'prediction': AE_preds}, save_path)

    return attack_score


acc = experiment_robustness(args, model, IFM=IFM, attack_name=re.sub(r'\d+','', attack_name), save_path=test_save_path)[0]

stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        logger.stats = pickle.load(f)

logger.add(category=cat, k=attack_name, v=acc, global_it=model_train_seed, unique=True)
logger.log('Adversarial {}: {:.5f}%'.format(attack_name, acc))

logger.save_stats('eval_stats.pkl')
logger.log('\nTesting completed.')
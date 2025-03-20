import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import copy
import torch
import torch.nn as nn
from core.attacks import create_attack, CWLoss
from core.utils import SmoothCrossEntropyLoss, ctx_noparamgrad_and_eval, set_bn_momentum, seed, CosineLR
from core.models import create_model
from core.metrics import accuracy
from method.criterion import MILoss
from method.loss import wscat_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(object):

    def __init__(self, info, args, logger):
        super(WATrainer, self).__init__()
        seed(args.seed)
        device = self.device = args.device
        self.logger = logger
        self.model = create_model(args=args, info=info, device=device, logger=logger)
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4*args.attack_iter,
                                         args.attack_step)
        self.num_classes = info['num_classes']
        self.base_dataset = info['data']

        args.feat_dim = self.model.feat_dim
        self.params = args
        self.init_optimizer(self.params.num_adv_epochs)
        self.criterion_contrastive = MILoss(args).to(device)

        param_groups = list(self.criterion_contrastive.parameters())
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optimizer.add_param_group(param_group)
        self.init_scheduler(self.params.num_adv_epochs)


    def init_optimizer(self, num_epochs):

        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups

        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay,
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return

    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'step':
            # scheduler for Wide-Resnet pepar
            milestones_frac, gamma = [0.3, 0.6, 0.8], 0.2
            milestones = []
            for i in milestones_frac:
                milestones.append(int(i * self.params.num_adv_epochs))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            if self.logger:
                self.logger.log(f'LR scheduler: step milestones-{milestones} gamma-{gamma}')

        elif self.params.scheduler == 'step_trades':
            # scheduler for TRADES
            milestones_frac, gamma = [0.75, 0.9], 0.1
            milestones = []
            for i in milestones_frac:
                milestones.append(int(i * self.params.num_adv_epochs))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  gamma=gamma, milestones=milestones)
            if self.logger:
                self.logger.log(f'LR scheduler: step-trades milestones-{milestones} gamma-{gamma}')

        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
            if self.logger:
                self.logger.log(f'LR scheduler: cosine max_lr-{self.params.lr} epochs-{int(num_epochs)}')

        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025,
                                                                 total_steps=int(num_epochs))
        elif self.params.scheduler == 'none':
            self.scheduler = None

        else:
            raise NotImplementedError(self.params.scheduler)


    def train(self, dataloader, epoch=0, adversarial=False, verbose=False, logger=None):

        metrics = pd.DataFrame()
        self.model.train()
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            update_iter += 1
            global_step = (epoch - 1) * len(dataloader) + update_iter
            self.update_steps = len(dataloader)
            self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
            if global_step == 1:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 2:
                set_bn_momentum(self.model, momentum=0.01)

            x, y = data
            x_l, y_l, x_u = self._data(x,y)
            del x, y

            if adversarial:
                assert self.params.beta >= 0
                con_ramp = sigmoid_rampup(global_step, 1, self.params.consistency_ramp_up * self.update_steps + 1)
                loss_dict, batch_metrics = self.wscat_loss(
                    x_l, y_l, x_u,
                    beta=self.params.beta, beta2=self.params.beta2,
                    consistency_cost=self.params.consistency_cost * con_ramp,)
            else:
                assert self.params.AT == 'standard'
                loss_dict, batch_metrics = self.standard_loss(x, y)

            loss = loss_dict['loss']
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_value_(self.model_parameters(), self.params.clip_grad)

            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            ema_update(self.wa_model, self.model, global_step,
                       decay_rate=self.params.tau if epoch <= self.params.consistency_ramp_up else self.params.tau_after,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            try:
                metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            except:
                metrics = metrics._append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            if logger is not None:
                for key in loss_dict:
                    logger.add("training", key, loss_dict[key].item(), global_step)
                # logger.log_info(global_step, ["training"])

        if self.params.scheduler not in ['cyclic', 'none']:
            self.scheduler.step()
        if logger is not None:
            logger.log_info(global_step, ["training"])
        update_bn(self.wa_model, self.model)
        return dict(metrics.mean())

    def _data(self, x, y):
        device = self.device
        idx = np.array(range(len(y)))
        idx_l, idx_u = idx[y != -1], idx[y == -1]
        if len(idx_u) > 0:
            x_u = x[idx_u]
            x_l, y_l = x[idx_l], y[idx_l]
            return x_l.to(device), y_l.to(device), x_u.to(device)
        else:
            return x.to(device), y.to(device), None

    def standard_loss(self, x, y):
        """
        Standard training.
        """

        self.optimizer.zero_grad()
        out = self.model(x)
        loss = SmoothCrossEntropyLoss(reduction='mean', smoothing=self.params.ls)(out, y)

        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        loss_dict = {'loss': loss}
        return loss_dict, batch_metrics

    def wscat_loss(self, x_l, y_l, x_u, beta, beta2, consistency_cost):

        loss, batch_metrics = wscat_loss(criterion_mi=self.criterion_contrastive,
                                         model=self.model, wamodel=self.wa_model,
                                         consistency_cost=consistency_cost,
                                         x_l=x_l, y=y_l, x_u=x_u, optimizer=self.optimizer,
                                         step_size=self.params.attack_step,
                                         epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter,
                                         beta=beta, beta2=beta2, attack=self.params.attack,
                                         contrast_label=self.params.contrast_label)

        return loss, batch_metrics


    def eval(self, dataloader, adversarial=False, verbose=True):
        acc = 0.0
        total = 0
        self.wa_model.eval()
        # for x, y in dataloader:
        device = self.device
        for data in tqdm(dataloader, desc='Eval : ', disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)
            total += x.size(0)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                with torch.no_grad():
                    out = self.wa_model(x_adv)
            else:
                with torch.no_grad():
                    out = self.wa_model(x)
            _, predicted = torch.max(out, 1)
            acc += (predicted == y).sum().item()
        acc /= total
        self.wa_model.train()
        return acc



    def save_model(self, path, epoch):

        torch.save({
            'model_state_dict': self.wa_model.state_dict(),
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'mi_embed1': self.criterion_contrastive.embed1.state_dict(),
            'mi_embed2': self.criterion_contrastive.embed2.state_dict(),
            'epoch': epoch
        }, path)

    def load_model(self, path, load_opt=True):

        checkpoint = torch.load(path)
        if 'mi_embed1' in checkpoint:
            self.criterion_contrastive.embed1.load_state_dict(checkpoint['mi_embed1'])
            self.criterion_contrastive.embed2.load_state_dict(checkpoint['mi_embed2'])
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']



def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked

def sigmoid_rampup(global_step, start_iter, end_iter):
    if global_step < start_iter:
        return 0.
    elif start_iter >= end_iter:
        return 1.
    else:
        rampup_length = end_iter - start_iter
        cur_ramp = global_step - start_iter
        cur_ramp = np.clip(cur_ramp, 0, rampup_length)
        phase = 1.0 - cur_ramp / rampup_length
        return np.exp(-5.0 * phase * phase)
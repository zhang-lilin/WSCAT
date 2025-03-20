import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.metrics import accuracy
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats
import numpy as np


def wscat_loss(criterion_mi, model, x_l, y, x_u, optimizer,
               step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, beta2=1.0,
               contrast_label='auto', attack='linf-pgd', label_smoothing=0.1,
               wamodel=None, consistency_cost=0., consistency_prop_label=0.,
               ):
    device = y.device
    num_y = y.size(0)
    if x_u is None:
        x_natural = x_l
        x_u = torch.empty(0,0)
    else:
        x_natural = torch.cat([x_l, x_u], dim=0)

    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    track_bn_stats(model, False)

    fea_nat, logits_natural = model(x_natural, feats=True)
    p_natural = F.softmax(logits_natural, dim=1).detach()

    if beta > 0:
        if contrast_label == 'auto':
            _, pseudo_labels = torch.max(logits_natural, 1)
            labels_4_contrast = pseudo_labels.detach()
            mask = None
        elif contrast_label == 'fixed':
            labels_4_contrast = y.detach()
            mask = None
        elif contrast_label:
            labels_4_contrast = None
            mask = None

        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                fea_adv, logits_adv = model(x_adv, feats=True)
                loss_atk = criterion_kl(F.log_softmax(logits_adv, dim=1), p_natural)
                if beta2 > 0:
                    loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss = loss_atk + beta2 * loss_con
                else:
                    loss = loss_atk
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()
    track_bn_stats(model, True)
    optimizer.zero_grad()

    fea_nat, logits_natural = model(x_natural, feats=True)
    loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
    if consistency_cost > 0 and x_u.size(0) > 0:
        softmax = torch.nn.Softmax(1)
        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
        loss_natural = loss_natural_sup + loss_consistency
    else:
        loss_consistency = torch.tensor(0)
        loss_natural = loss_natural_sup

    if beta > 0:
        fea_adv, logits_adv = model(x_adv, feats=True)
        loss_adv = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
        if beta2 > 0:
            loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
        else:
            loss_con = torch.tensor(0)
        batch_metrics = {
            'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
            'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
    else:
        loss_adv = loss_con = torch.tensor(0)
        batch_metrics = {
            'clean_acc': accuracy(y, logits_natural[:num_y].detach())}

    loss_robust = loss_adv + beta2 * loss_con
    loss = loss_natural + beta * loss_robust

    loss_dict = {
        'loss': loss,
        'nat_loss': loss_natural_sup,
        'unsup_loss': loss_consistency,
        'rob_loss': loss_robust,
        'adv_loss': loss_adv,
        'con_loss': loss_con,
    }

    return loss_dict, batch_metrics

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

class InformativeFeaturePackage(nn.Module):
    def __init__(self, model, device, eps=0.03, attack_iter=10, IFD_iter=200, IFD_lr=0.1):
        super(InformativeFeaturePackage, self).__init__()
        self.model = model
        self.device = device

        # PGD-based IFD attack hyper-parameter
        self.eps = eps
        self.attack_iter = attack_iter
        self.alpha = self.eps/attack_iter*2.3
        self.eta = 1e-2

        # IFD hyper-parameter
        self.IFD_iter = IFD_iter
        self.IFD_lr = IFD_lr
        self.cw_c = 0.1
        self.pgd_c = 10
        self.beta = 0.3
        self.grad = 1

        # define loss
        self.mse = nn.MSELoss(reduction='none')

        # softplus
        self.softplus = nn.Softplus()

    @staticmethod
    def grad_on_off(model, switch=False):
        for param in model.parameters():
            param.requires_grad=switch

    @staticmethod
    def kl_div(p, lambda_r):
        delta = 1e-10
        p_var = p.var(dim=[2, 3])
        q_var = (lambda_r.squeeze(-1).squeeze(-1)) ** 2

        eq1 = p_var / (q_var + delta)
        eq2 = torch.log((q_var + delta) / (p_var + delta))

        kld = 0.5 * (eq1 + eq2 - 1)

        return kld.mean()

    @staticmethod
    def sample_latent(latent_r, lambda_r, device):
        eps = torch.normal(0, 1, size=lambda_r.size()).to(device)
        return latent_r + lambda_r.mul(eps)


    def sample_robust_and_non_robust_latent(self, latent_r, lambda_r):

        var = lambda_r.square()
        r_var = latent_r.var(dim=(2,3)).view(-1)

        index = (var > r_var.max()).float()
        return index



    def find_features(self, input, labels, pop_number, forward_version=False):

        latent_r = self.model.rf_output(input, pop=pop_number)
        lambda_r = torch.zeros([*latent_r.size()[:2],1,1]).to(self.device).requires_grad_()
        optimizer = torch.optim.Adam([lambda_r], lr=self.IFD_lr)

        for i in range(self.IFD_iter):

            lamb = self.softplus(lambda_r)
            latent_z = self.sample_latent(latent_r.detach(), lamb, self.device)

            outputs = self.model.rf_output(latent_z.clone(), intermediate_propagate=pop_number)
            kl_loss = self.kl_div(latent_r.detach(), lamb)
            ce_loss = F.cross_entropy(outputs, labels)
            loss = ce_loss + self.beta * kl_loss
            # print(f'loss:{loss.item()} ce:{ce_loss.item()} kl:{kl_loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        robust_lambda_r = lambda_r.clone().detach()

        # robust and non-robust index
        robust_index  = self.sample_robust_and_non_robust_latent(latent_r, self.softplus(robust_lambda_r))
        non_robust_index = 1-robust_index

        # robust and non-robust feature
        robust_latent_z     = latent_r * robust_index
        non_robust_latent_z = latent_r * non_robust_index

        robust_outputs = self.model.rf_output(robust_latent_z.clone(), intermediate_propagate=pop_number).detach()
        _, robust_predicted = robust_outputs.max(1)

        non_robust_outputs = self.model.rf_output(non_robust_latent_z.clone(), intermediate_propagate=pop_number).detach()
        _, non_robust_predicted = non_robust_outputs.max(1)

        if forward_version:
            return latent_r, robust_latent_z, non_robust_latent_z, \
                   robust_predicted, non_robust_predicted, robust_index, non_robust_index
        return latent_r, robust_latent_z, non_robust_latent_z, robust_predicted, non_robust_predicted

    @staticmethod
    def tanh_space(x):
        return 1 / 2 * (torch.tanh(x) + 1)

    @staticmethod
    def inverse_tanh_space(x):
        return 0.5 * torch.log((1 + x*2-1) / (1 - (x*2-1)))

    # def NRF(self, images, labels):
    #
    #     images = images.clone().detach().to(self.device)
    #     labels = labels.clone().detach().to(self.device)
    #
    #     _, _, _, \
    #     _, _, _, non_robust_index \
    #         = self.find_features(images, labels, pop_number=3, forward_version=True)
    #
    #     CE = nn.CrossEntropyLoss()
    #     dim = len(images.shape)
    #
    #     epsilon = self.eps
    #     device = self.device
    #
    #     adv_images = images.detach() + torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(device).detach()
    #     adv_images = torch.clamp(adv_images, 0.0, 1.0)
    #     for step in range(self.attack_iter):
    #         adv_images.requires_grad_()
    #         with torch.enable_grad():
    #             latent_r = self.model.rf_output(adv_images, pop=3)
    #             outputs = self.model.rf_output(latent_r.clone(), intermediate_propagate=3)
    #             f_loss = self.f(outputs, labels, self.device).sum()
    #             grad_latent = torch.autograd.grad(-f_loss, latent_r,
    #                                               retain_graph=True, create_graph=False)[0]
    #             cost_NR = torch.dist(latent_r, latent_r.detach() - non_robust_index * grad_latent.detach()) # something new method
    #             ce_loss = CE(outputs, labels)
    #             cost = ce_loss - self.grad*cost_NR
    #         grad = torch.autograd.grad(cost, [adv_images])[0]
    #         adv_images = adv_images.detach() + 1/255 * torch.sign(grad.detach())
    #         adv_images = torch.min(torch.max(adv_images, images - epsilon), images + epsilon)
    #         adv_images = torch.clamp(adv_images, 0.0, 1.0)
    #
    #     best_adv_images = Variable(torch.clamp(adv_images, 0.0, 1.0), requires_grad=False)
    #     return best_adv_images

    def NRF(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        _, _, _, \
        _, _, _, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model.rf_output(adv_images, pop=3)
            outputs = self.model.rf_output(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels, self.device).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_NR = torch.dist(latent_r, latent_r.detach() - non_robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss - self.grad*cost_NR

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            # best_adv_images = torch.min(torch.max(best_adv_images, images - 8/255), images + 8/255)
            # best_adv_images = torch.clamp(best_adv_images, 0.0, 1.0)

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def NRF2(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        _, _, _, \
        _, _, robust_index, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model.rf_output(adv_images, pop=3)
            outputs = self.model.rf_output(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels, self.device).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_NR = torch.dist(latent_r, latent_r.detach() - non_robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss + self.grad*cost_NR

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def RF(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        _, _, _, \
        _, _, robust_index, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model.rf_output(adv_images, pop=3)
            outputs = self.model.rf_output(latent_r.clone().to(self.device), intermediate_propagate=3)
            f_loss = self.f(outputs, labels, self.device).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_R = torch.dist(latent_r, latent_r.detach() - robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss - self.grad*cost_R

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def RF2(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        _, _, _, \
        _, _, robust_index, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model.rf_output(adv_images, pop=3)
            outputs = self.model.rf_output(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels, self.device).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_R = torch.dist(latent_r, latent_r.detach() - robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss + self.grad*cost_R

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images
    # f-function in the paper
    @staticmethod
    def f(outputs, labels, device):
        one_hot_labels = torch.eye(len(outputs[0])).to(device)[labels]

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(-1 * (i - j), min=0)


    def forward(self, input, labels, pop_number, noise=0):

        latent_r, robust_latent_z, non_robust_latent_z, \
        robust_predicted, non_robust_predicted, robust_index, non_robust_index \
            = self.find_features(input, labels, pop_number=pop_number, forward_version=True)

        robust_noise_output = self.model.rf_output((robust_latent_z+robust_index*noise*torch.randn(robust_latent_z.shape).to(self.device)).clone(), intermediate_propagate=pop_number)
        non_robust_noise_output = self.model.rf_output((non_robust_latent_z+(1-robust_index)*noise*torch.randn(non_robust_latent_z.shape).to(self.device)).clone(), intermediate_propagate=pop_number)

        _, robust_noise_predicted = robust_noise_output.max(1)
        _, non_robust_noise_predicted = non_robust_noise_output.max(1)

        return robust_predicted, non_robust_predicted, robust_noise_predicted, non_robust_noise_predicted

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import os
from tensorboardX import SummaryWriter
import numpy as np
import torch.optim as optim
import warnings
from tqdm import tqdm
from torch.nn.functional import mse_loss
import random
from torchvision import transforms
from kornia import augmentation
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as sp
import torch.backends.cudnn as cudnn
from advertorch.attacks import LinfBasicIterativeAttack, PGDAttack
from nets import Generator_2
from utils import ScoreLoss, ImagePool, MultiTransform, reset_model, get_dataset, cal_prob, cal_label, setup_seed, \
    get_model, print_log, test, test_robust, save_checkpoint
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # "0,1"#


def info_entropy(x):
    return torch.distributions.Categorical(probs=torch.softmax(x, dim=-1)).entropy()



class InfoEntropyLoss(nn.Module):
    def __init__(self):
        super(InfoEntropyLoss, self).__init__()
        pass

    def forward(self, x_logits, num_class=10):
        x_logits -= x_logits.min(dim=1).values.view(-1, 1).detach()
        x_logits /= x_logits.max(dim=1).values.view(-1, 1).detach()
        x_logits = x_logits.sum(0) / x_logits.size(0)

        return -(((x_logits + 1e-10).log()) * x_logits).sum(
            -1) / num_class  # torch.distributions.Categorical(probs=correct_batch).entropy()/num_class #torch.softmax(correct_batch, dim=-1)

class DiverseEntropyLoss(nn.Module):
    def __init__(self):
        super(DiverseEntropyLoss, self).__init__()
        pass

    def forward(self, ChannelNoiseMatixs, targets, criticize=False):
        loss = 0
        targets = targets.squeeze()
        y_list = torch.unique(targets, sorted=True)
        targets = torch.searchsorted(y_list, targets)
        targets = torch.zeros(targets.size(0), y_list.size(0), device=targets.device).scatter_(1,
                                                                                               targets.reshape(-1, 1),
                                                                                               1)  # NxL
        cri = 0
        for mtx in ChannelNoiseMatixs:
            mtx = mtx.flatten(start_dim=1)  # N x F

            mtx = mtx / mtx.norm(dim=-1, keepdim=True)  # 向量正则化
            # mtx = -mtx * torch.log2(torch.abs(mtx) + 1e-8)
            # mtx = F.softmax(mtx,dim=-1)
            mtx = -mtx * torch.log2(torch.abs(mtx) + 1e-12)  # 1e-8

            if criticize:
                cri += mtx.sum(dim=-1, keepdim=True)
            loss_info = targets.t() @ mtx / (targets.t().sum(1).view(-1, 1))  # L x F

            loss += (loss_info.sum() / y_list.size(0))  # y_list.size(0)
        if criticize:
            return cri
        return loss / len(ChannelNoiseMatixs)


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class InterDifferenceLoss(nn.Module):
    def __init__(self):
        super(InterDifferenceLoss, self).__init__()
        pass

    def forward(self, ChannelNoiseMatixs, targets):
        loss = 0
        targets = targets.squeeze()
        y_list = torch.unique(targets, sorted=True)
        targets = torch.searchsorted(y_list, targets)
        targets = torch.zeros(targets.size(0), y_list.size(0), device=targets.device).scatter_(1,
                                                                                               targets.reshape(-1, 1),
                                                                                               1)  # NxL
        y_eyes = torch.eye(y_list.size(0), device=targets.device)
        for mtx in ChannelNoiseMatixs:
            mtx = mtx.flatten(start_dim=1)  # N x F

            mtx = targets.t() @ mtx / (targets.t().sum(1).view(-1, 1))
            mtx = normalize(mtx)
            CosineSimilarity = torch.mm(mtx, mtx.permute(1, 0))
            CosineDistance = F.mse_loss(torch.triu(CosineSimilarity), torch.triu(y_eyes))
            loss += CosineDistance.sum()
        return loss


class Synthesizer():
    def __init__(self, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 sample_batch_size, save_dir, dataset):
        super(Synthesizer, self).__init__()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.score_loss = ScoreLoss()
        self.num_classes = num_classes
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.dataset = dataset
        self.infoLoss = DiverseEntropyLoss()
        self.interDiffLoss = InterDifferenceLoss()
        self.ceLoss = nn.CrossEntropyLoss()  # label_smoothing=0.1
        self.generator = generator.cuda().train()
        self.interSimilarLoss = InterSimilarityLoss()
        self.intraDiversity = IntraDiversityLoss()

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

    def get_data(self, dataset):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        if dataset == 'tiny':
            batch_size = 256  # 512
        else:
            batch_size = 256
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, )
        return self.data_loader

    def gen_data(self, student, epoch, dataset):
        student.eval()
        # best_cost = 1e6#*torch.ones(self.sample_batch_size,1).cuda()#1e6
        best_ce_cost = 1e6
        best_is_cost = 1e6
        # best_inputs = None
        best_ce_inputs = None
        best_is_inputs = None
        entropy_weight = torch.tensor(0.001, requires_grad=True)

        z = torch.randn(size=(self.sample_batch_size, self.nz)).cuda()  #
        z.requires_grad = True

        label_z = torch.randn(size=(self.sample_batch_size, self.nz)).cuda()  #
        label_z.requires_grad = True

        # if dataset == 'tiny':
        #    targets = torch.arange(0, self.num_classes).repeat(self.sample_batch_size//self.num_classes)
        # else:
        targets = torch.randint(low=0, high=self.num_classes, size=(self.sample_batch_size,))
        targets = targets.sort()[0]
        targets = targets.cuda()
        avg_data = 0
        all_loss = None
        reset_model(self.generator)
        # if dataset in ['cifar10', 'svhn']:#['cifar10', 'svhn']
        #    lr = self.lr_g*0.3 if epoch in [int(args.epochs*0.3),int(args.epochs*0.5)] else self.lr_g
        if dataset == 'svhn':
            lr = self.lr_g * 0.3 if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5)] else self.lr_g
        elif dataset == 'cifar10':
            lr = self.lr_g * 0.3 if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5)] else self.lr_g
        elif dataset == 'cifar100':
            lr = self.lr_g * 0.3 if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5)] else self.lr_g
        # elif dataset =='tiny':
        # lr = self.lr_g*0.1 if epoch  in [50,150] else self.lr_g
        #    lr = self.lr_g * 0.3 if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5)] else self.lr_g
        elif dataset == 'mnist':
            lr = self.lr_g * 0.3 if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5)] else self.lr_g
        elif dataset == 'fmnist':
            lr = self.lr_g * 0.3 if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5)] else self.lr_g
        else:
            lr = self.lr_g
        para_ce = 1
        para_info = 2
        para_adv = 1.2
        if dataset in ['cifar100']:
            para_info = 5  # 3# best:5
            para_adv = 1.2  # 1.8#para_info*0.6 # best:1.2
        elif dataset in ['tiny']:
            para_info = 6  # 6#6#3# best:5
            para_adv = 1.4  # 1#1.2#1.8#para_info*0.6 # best:1.2
        optimizer = torch.optim.Adam(self.generator.parameters(), lr, betas=[0.5, 0.999],
                                     weight_decay=0)  # , weight_decay=0
        targets_vec = torch.zeros(targets.shape[0], num_class, device=targets.device).scatter_(1,
                                                                                               targets.reshape(-1, 1),
                                                                                               1)
        for it in range(self.iterations):
            optimizer.zero_grad()
            inputs_ = self.generator(z, targets_vec, label_z)  # bs,nz

            inputs = inputs_[0]
            input_fs = inputs_[1:]
            global_view, _ = self.aug(inputs)  # crop and normalize
            s_out = student(global_view)
            loss_ce = self.score_loss(s_out, targets)  # ce_loss
            loss_info = self.infoLoss(input_fs, targets)
            loss_id = self.intraDiversity(F.softmax(s_out, dim=-1), targets)  #

            loss = para_ce * loss_ce - para_info * loss_info + para_adv * loss_id

            loss_item = [loss_ce, loss_info, loss_id]
            if best_is_cost > loss.item() or best_is_inputs is None:
                best_is_cost = loss.item()
                all_loss = loss_item
                best_is_inputs = inputs.data

            loss.backward()
            optimizer.step()

        print("Total loss:{:.4}, loss_ce:{:.4}, loss_info:{:.4}, loss_id:{:.4} \n".format(best_is_cost, all_loss[0],
                                                                                          all_loss[1],
                                                                                          all_loss[2]))  #:.4
        best_inputs = best_is_inputs  # (best_ce_inputs+best_is_inputs) /2
        # save best inputs and reset data iter
        self.data_pool.add(best_inputs)  # 生成了一个batch的数据


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--score', type=float, default=0,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    # Data Free

    parser.add_argument('--save_dir', default='run/mnist', type=str)

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--g_steps', default=30, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="score", type=str,
                        help='score or label')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    args = parser.parse_args()
    return args


def loss_s_main(loss_ce, loss_mse, score_val):
    return loss_ce + loss_mse * score_val


def loss_s_bd(substitute_outputs, original_score, score_val):
    substitute_score = F.softmax(substitute_outputs, dim=1)
    label = original_score.max(1)[1]
    idx = torch.where(substitute_outputs.max(1)[1] != label)[0]

    return F.cross_entropy(substitute_outputs[idx], label[idx]) + score_val * mse_loss(substitute_score[idx],
                                                                                       original_score[idx],
                                                                                       reduction='mean')


def loss_s_adv(images, models, label, dataset, score_val):
    if dataset == "mnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "cifar10" or dataset == "cifar100":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)
    elif dataset == "fmnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "svhn" or dataset == "tiny":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)

    sub_net, blackBox_net = models
    # adversary_ghost = LinfBasicIterativeAttack(
    #     sub_net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    #     nb_iter=100, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    #     targeted=False)
    adversary_ghost = PGDAttack(
        sub_net,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=cfgs['test_epsilon'],
        nb_iter=3, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
        targeted=False)  # 20
    adv_inputs_ghost = adversary_ghost.perturb(images, label)
    original_score = cal_prob(blackBox_net, images.detach())  # prob
    sub_score = cal_prob(sub_net, images)  # prob
    sub_adv_score = cal_prob(sub_net, adv_inputs_ghost.detach())  # prob

    label = original_score.max(1)[1]
    idx = \
    torch.where((sub_score.max(1)[1] != sub_adv_score.max(1)[1]) & (sub_score.max(1)[1] != original_score.max(1)[1]))[0]
    return F.cross_entropy(sub_score[idx], label[idx])


class RelationLoss(nn.Module):
    def __init__(self):
        super(RelationLoss, self).__init__()
        pass

    def forward(self, substitute_score, targets, score=0):
        if not score:
            n_cls = substitute_score.size(-1)
            targets = targets.squeeze()
            targets = torch.zeros(targets.size(0), n_cls, device=targets.device).scatter_(1, targets.reshape(-1, 1),
                                                                                          1)  # NxL

        substitute_score = (substitute_score.unsqueeze(-1) - substitute_score.unsqueeze(-2)).abs().flatten(start_dim=1)
        targets = (targets.unsqueeze(-1) - targets.unsqueeze(-2)).abs().flatten(start_dim=1)
        loss = 1 - F.cosine_similarity(substitute_score, targets)
        loss = loss.sum() / loss.size(0)

        return loss


def select_or_exclude_labelcol(tensor, label, exclude=True):
    # 将标签数组转换为重复的二维张量
    label_tensor = label.repeat(tensor.size(1), 1).t()

    # 使用masked_select选择不等于对应标签的列
    tmp = torch.arange(tensor.size(1), device=tensor.device)
    if exclude:
        condition = (label_tensor != tmp.unsqueeze(0))
    else:
        condition = (label_tensor == tmp.unsqueeze(0))
    result = torch.masked_select(tensor, condition)

    # 将结果重新整形为2D张量
    return result.view(tensor.size(0), -1)


def block_min_max(tensor, blocks, mode='max'):
    # 创建一个与tensor相同大小的索引张量，并在每个块的位置上用该块的ID填充
    block_indices = torch.zeros_like(tensor)
    # print(blocks.shape,tensor.shape)
    # exit()
    block_indices.scatter_(1, blocks.unsqueeze(1).expand_as(tensor), 1)
    one_tensor = torch.ones_like(block_indices, device=tensor.device)

    if mode == 'max':
        tensor_max = tensor.max(dim=1, )[0].unsqueeze(-1)
    else:
        tensor_max = tensor.min(dim=1, )[0].unsqueeze(-1)
    tensor_max = block_indices * tensor_max
    # print(tensor_max)
    if mode == 'max':
        tensor_max = torch.where(block_indices == 0, -torch.inf * one_tensor, tensor_max)

        tensor_max = tensor_max.max(dim=0)[0].unsqueeze(0)
    else:
        tensor_max = torch.where(block_indices == 0, torch.inf * one_tensor, tensor_max)
        tensor_max = tensor_max.min(dim=0)[0].unsqueeze(0)

    tensor_max = (tensor_max * block_indices).sum(dim=1, keepdim=True)
    return tensor_max


def weight_energy_distance(target_energy, non_target_energies):
    # 根据目标类能量距离对非目标类能量距离进行调整
    # adjusted_non_target_probs = -target_energy *non_target_energies.log()
    adjusted_non_target_probs = -target_energy * torch.log(non_target_energies + 1e-12)  # non_target_energies.log()
    adjusted_non_target_probs = F.log_softmax(adjusted_non_target_probs, dim=1)

    return adjusted_non_target_probs


class IntraDiversityLoss(nn.Module):
    def __init__(self):
        super(IntraDiversityLoss, self).__init__()
        pass

    def forward(self, substitute_score, targets, criticize=False):
        ############ 选出正确分类的样本 ##################
        idx_false = torch.where(substitute_score.max(1)[1] != targets)[0]
        #  # 方案2 p->op->mean->std
        ############### 从距离的角度来考虑该问题，即概率p越大,与该类的距离越小，该距离可以简化为1-p,这样考虑不仅考虑了对gt类的距离，也考虑了数据边界的高维性，相对来说更合理################################
        out_score_exclude = select_or_exclude_labelcol(substitute_score, targets, exclude=True)
        out_score_target = select_or_exclude_labelcol(substitute_score, targets, exclude=False)

        out_score_target[idx_false] = 1  # If misclassification, don't consider its P_target, because its P_target is not meaningful at this time

        # norm according to distance thought
        norm_mtx = weight_energy_distance(out_score_target, out_score_exclude)

        if criticize:
            idx_ex = torch.where(substitute_score.max(1)[1] != targets)[0]
            std_o = torch.std(norm_mtx, dim=1, keepdim=True)
            std_o[idx_ex] = 0
            return std_o

        # norm_mtx = norm_mtx[idx]
        # targets = targets[idx]
        y_list = torch.unique(targets, sorted=True)
        targets_ = torch.searchsorted(y_list, targets)
        targets = torch.zeros(targets_.size(0), y_list.size(0), device=targets_.device).scatter_(1,targets_.reshape(-1,1),1)  # NxL

        out_t_score = targets.t() @ norm_mtx  # L x F
        out_t_score = torch.std(out_t_score, dim=1) / (targets.t().sum(1).view(-1, 1))

        loss = (out_t_score.sum() / y_list.size(0))  # y_list.size(0)

        return loss  # +diff_loss#+loss_main


def cosine_similarity(matrix):
    norms = torch.norm(matrix, dim=1, keepdim=True)
    normalized_matrix = matrix / norms
    similarity_matrix = torch.mm(normalized_matrix, normalized_matrix.t())
    ######### scale into [0, 1] #############
    similarity_matrix = (similarity_matrix + 1) / 2
    return similarity_matrix


class InterSimilarityLoss(nn.Module):
    def __init__(self):
        super(InterSimilarityLoss, self).__init__()
        pass

    def forward(self, substitute_score, targets, criticize=False):
        ############ 直觉上不同类之间的P_t - P_i 的相似性可以拉近类间相似性 ##################

        out_score_exclude = select_or_exclude_labelcol(substitute_score, targets, exclude=True)
        out_score_target = select_or_exclude_labelcol(substitute_score, targets, exclude=False)

        max_out = out_score_exclude.log().max(dim=1, keepdim=True)[0]  # .unsqueeze(0)
        mask = torch.gt(out_score_target - max_out, 0.2).float()
        diff = out_score_target * mask
        # norm according to distance thought

        loss = diff.sum() / mask.sum()  # y_list.size(0)

        return loss

def kd_train(synthesizer, model, optimizer, dataset, score_val):
    sub_net, blackBox_net = model
    sub_net.train()
    blackBox_net.eval()
    # with tqdm(synthesizer.get_data()) as epochs:
    data = synthesizer.get_data(dataset)
    for idx, (images) in enumerate(data):
        optimizer.zero_grad()
        images = images.cuda()
        original_score = cal_prob(blackBox_net, images)  # prob
        substitute_outputs = sub_net(images.detach())
        substitute_score = F.softmax(substitute_outputs, dim=1)
        loss_mse = mse_loss(
            substitute_score, original_score, reduction='mean')
        label = cal_label(blackBox_net, images)  # label
        loss_ce = F.cross_entropy(substitute_outputs, label)  # ,label_smoothing=0.1

        loss_main = loss_s_main(loss_ce, loss_mse, score_val)  # loss_ce + loss_mse * score_val

        loss = loss_main  # + loss_adv  # 0.2*+ loss_relation#+ loss_adv
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    dir = './saved/ours'
    if not os.path.exists(dir):
        os.mkdir(dir)

    args = args_parser()
    setup_seed(args.seed)
    train_loader, test_loader = get_dataset(args.dataset)

    public = dir + '/logs_{}_{}'.format(args.dataset, str(args.score))
    if not os.path.exists(public):
        os.mkdir(public)
    log = open('{}/log_ours.txt'.format(public), 'w')

    list = [i for i in range(0, len(test_loader.dataset))]
    data_list = random.sample(list, 1024)
    val_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=128,
                                             sampler=sp.SubsetRandomSampler(data_list), num_workers=4)

    tf_writer = SummaryWriter(log_dir=public)
    sub_net, _ = get_model(args.dataset, 0)

    blackBox_net, state_dict = get_model(args.dataset, 1)
    blackBox_net.load_state_dict(state_dict)

    print_log("===================================== \n", log)
    acc, _ = test(blackBox_net, val_loader)
    print_log("Accuracy of the black-box model:{:.4} % \n".format(acc), log)
    acc, _ = test(sub_net, val_loader)
    print_log("Accuracy of the substitute model:{:.4} % \n".format(acc), log)
    asr, val_acc = 0.0, 0.0  # test_robust(val_loader, sub_net, blackBox_net, args.dataset)
    print_log("ASR:{:.3} %, val acc:{:.4} % \n".format(asr, val_acc), log)
    print_log("===================================== \n", log)
    log.flush()

    ################################################
    # data generator
    ################################################
    nz = args.nz
    nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" or args.dataset == "tiny" else 1
    # img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" else 28

    if "cifar" in args.dataset or args.dataset == "svhn":
        img_size = 32
    elif "mnist" in args.dataset:
        img_size = 28
    elif args.dataset == "tiny":
        img_size = 64

    if "cifar" in args.dataset or args.dataset == "svhn":
        img_size2 = (3, 32, 32)
    elif "mnist" in args.dataset:
        img_size2 = (1, 28, 28)
    elif args.dataset == "tiny":
        img_size2 = (3, 64, 64)

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "tiny":
        num_class = 200
    else:
        num_class = 10

    generator = Generator_2(nz=nz, ngf=64, img_size=img_size, nc=nc, num_class=num_class).cuda()  # fix

    # ====================
    sub_net = torch.nn.DataParallel(sub_net)
    blackBox_net = torch.nn.DataParallel(blackBox_net)
    generator = torch.nn.DataParallel(generator)
    # ====================

    args.cur_ep = 0
    # img_size2 = (
    #     3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" else (1, 28, 28)

    synthesizer = Synthesizer(generator,
                              nz=nz,
                              num_classes=num_class,
                              img_size=img_size2,
                              iterations=args.g_steps,
                              lr_g=args.lr_g,
                              sample_batch_size=args.batch_size,
                              save_dir=args.save_dir,
                              dataset=args.dataset)
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if args.dataset in ['fmnist', 'mnist']:  # ['cifar10', 'svhn', 'fmnist']:
        weight_decay = 5e-3  ##5e-4
    elif args.dataset in ['cifar10']:  # ['cifar10', 'svhn', 'fmnist']:
        weight_decay = 5e-4  # 5e-4
    elif args.dataset in ['cifar100']:  # ['cifar10', 'svhn', 'fmnist']:
        weight_decay = 5e-4  # best: 5e-4##5e-4
    elif args.dataset in ['tiny']:  # ['cifar10', 'svhn', 'fmnist']:
        weight_decay = 5e-4  # 0#5e-4#5e-4#5e-4
    elif args.dataset in ['svhn']:  # ['cifar10', 'svhn', 'fmnist']:
        weight_decay = 5e-3  # 2e-3#5e-4
    else:
        weight_decay = 0
    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)
    sub_net.train()
    best_acc = -1
    best_asr = -1
    scheduler = None
    if args.dataset in ['cifar10', 'cifar100']:  # , 'cifar100','tiny', 'cifar100'['cifar10', 'svhn']
        scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.3)

    best_acc_ckpt = '{}/{}_ours_acc.pth'.format(public, args.dataset)
    best_asr_ckpt = '{}/{}_ours_asr.pth'.format(public, args.dataset)
    for epoch in tqdm(range(args.epochs)):
        # 1. Data synthesis
        synthesizer.gen_data(sub_net, epoch, args.dataset)  # g_steps
        kd_train(synthesizer, [sub_net, blackBox_net], optimizer, args.dataset, args.score)
        if args.dataset in ['cifar10', 'cifar100']:  # , 'cifar100','tiny'#,'cifar100'
            scheduler.step()

        if epoch % 1 == 0:  # 250*40, 250*10=2.5k
            acc, test_loss = test(sub_net, val_loader)
            asr, val_acc = test_robust(val_loader, sub_net, blackBox_net, args.dataset)

            save_checkpoint({
                'state_dict': sub_net.state_dict(),
                'epoch': epoch,
            }, acc > best_acc, best_acc_ckpt)
            #
            save_checkpoint({
                'state_dict': sub_net.state_dict(),
                'epoch': epoch,
            }, asr > best_asr, best_asr_ckpt)

            best_asr = max(best_asr, asr)
            best_acc = max(best_acc, acc)

            print_log("Accuracy of the substitute model:{:.4} %, best accuracy:{:.4} % \n".format(acc, best_acc), log)
            print_log("ASR:{:.4} %, best asr:{:.4} %, val acc:{:.4} % \n".format(asr, best_asr, val_acc), log)
            log.flush()



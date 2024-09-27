import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import numba
from numba import jit
import gc
import matplotlib.pyplot as plt
import random
from multiprocessing import Process
import torch
import torch.optim as optim
import torch.nn as nn


from dataloader import pallmediaMyDataset, _collate_fn, cutpallmediaMyDataset
import model
import allmediaresult

from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")


def train_model(
    model, train_loader, epoch, num_epochs, optimizer, current_lr, log_every, device
):
    """
    训练模型函数

    Args:
        model (torch.nn.Module): 要训练的模型
        train_loader (torch.utils.data.DataLoader): 训练数据的数据加载器
        epoch (int): 当前训练的轮数
        num_epochs (int): 总共的训练轮数
        optimizer (torch.optim.Optimizer): 优化器
        current_lr (float): 当前的学习率
        log_every (int): 每隔多少步打印一次日志
        device (str): 训练设备，例如 'cuda' 或 'cpu'

    Returns:
        train_loss_epoch (float): 当前轮的平均训练损失
        train_auc_epoch (float): 当前轮的平均训练 AUC
        train_acc_epoch (float): 当前轮的平均训练准确率
        y_pre (list): 当前轮的预测标签
        y_scores (list): 当前轮的预测分数
        y_trues (list): 当前轮的真实标签
    """

    _ = model.train()
    model.cuda()
    # model.to(device)
    y_preds = []
    y_trues = []
    y_pre = []
    losses = []
    output_list, labels_list, pre_list = [], [], []
    total_loss = 0
    index = 0

    for i, (image1, label, bz) in enumerate(train_loader):
        optimizer.zero_grad()
        image1 = image1.cuda()
        label = label.cuda()
        # weight=weight.cuda()
        # bz=bz.cuda()

        prediction = model.forward(image1.float(), bz, device)
        # prediction=prediction.cuda()

        # loss_fn = torch.nn.CrossEntropyLoss(weight=weight[0])
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn=loss_fn.cuda()
        loss = loss_fn(prediction, label)
        loss.backward()
        total_loss += float(loss.item())
        index += 1
        optimizer.step()

        # loss_value = loss.item()
        # losses.append(loss_value)

        probas = torch.softmax(prediction, dim=1)
        pre = torch.argmax(probas, dim=1)

        probas = probas.cpu()
        pre = pre.cpu()
        label = label.cpu()
        output_list.extend(probas.detach().numpy())
        y_trues.extend(label.detach().numpy())
        pre_list.extend(pre.detach().numpy())

    y_scores = [o[1] for o in output_list]
    y_pre = pre_list
    auc = metrics.roc_auc_score(y_trues, y_scores)
    acc = metrics.accuracy_score(y_trues, y_pre)

    train_loss_epoch = np.round(total_loss / index, 4)
    train_auc_epoch = np.round(auc, 4)
    train_acc_epoch = np.round(acc, 4)
    return train_loss_epoch, train_auc_epoch, train_acc_epoch, y_pre, y_scores, y_trues


def evaluate_model(model, val_loader, device):
    """
    评估模型函数

    Args:
        model (torch.nn.Module): 要评估的模型
        val_loader (torch.utils.data.DataLoader): 验证数据的数据加载器
        device (str): 评估设备，例如 'cuda' 或 'cpu'

    Returns:
        val_loss_epoch (float): 当前轮的平均验证损失
        val_auc_epoch (float): 当前轮的平均验证 AUC
        val_acc_epoch (float): 当前轮的平均验证准确率
        y_pre (list): 当前轮的预测标签
        y_scores (list): 当前轮的预测分数
        y_trues (list): 当前轮的真实标签
        out (list): 当前轮的预测输出

    """
    _ = model.eval()
    model.cuda()
    # model.to(device)

    y_trues = []
    y_preds = []
    y_pre = []
    losses = []
    output_list, labels_list, pre_list = [], [], []
    out = []
    total_loss, index = 0, 0
    with torch.no_grad():

        for i, (image1, label, bz) in enumerate(val_loader):
            image1 = image1.cuda()
            # image2 = image2.to(device)
            label = label.cuda()
            # bz = bz.cuda()
            # weight=weight.cuda()

            prediction = model.forward(image1.float(), bz, device)
            # prediction=prediction.cuda()
            out.append(prediction)

            # loss_fn = torch.nn.CrossEntropyLoss(weight=weight[0])
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn=loss_fn.cuda()
            loss = loss_fn(prediction, label)
            total_loss += loss.item()
            index += 1
            # loss_value = loss.item()
            # losses.append(loss_value)

            probas = torch.softmax(prediction, dim=1)
            pre = torch.argmax(probas, dim=1)

            probas = probas.cpu()
            pre = pre.cpu()
            label = label.cpu()
            output_list.extend(probas.detach().numpy())
            y_trues.extend(label.detach().numpy())
            pre_list.extend(pre.detach().numpy())

        y_scores = [o[1] for o in output_list]
        y_pre = pre_list
        auc = metrics.roc_auc_score(y_trues, y_scores)
        acc = metrics.accuracy_score(y_trues, y_pre)

        val_loss_epoch = np.round(total_loss / index, 4)
        val_auc_epoch = np.round(auc, 4)
        val_acc_epoch = np.round(acc, 4)
    return val_loss_epoch, val_auc_epoch, val_acc_epoch, y_pre, y_scores, y_trues, out


def get_lr(optimizer):
    """
    获取当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def allmediarun(
    args, prefix_name, round, cuda, mrnet, imgpath, samplepath, batch_size, lr
):
    """
    函数用于对所有视频进行训练和验证模型，得到视频级的结果。

    args: 传入的参数集合。
    prefix_name: 前缀名称，用于指定输出文件夹的名称。
    round: 当前的训练轮次。
    cuda: 使用的 CUDA 设备编号。
    mrnet: 训练的神经网络模型。
    imgpath: 图像数据的路径。
    samplepath: 样本数据的路径。
    batch_size: 批处理大小。
    lr: 学习率。
    """

    # 检查并创建输出和模型保存的目录
    if not os.path.exists(
        f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}"
    ):
        os.makedirs(f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}")
    else:
        # 如果目录已存在，清空其中的文件
        files = os.listdir(
            f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}"
        )
        for file in files:
            os.remove(
                f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}/{file}"
            )
    # 检查并创建模型保存的目录
    if not os.path.exists(
        f"{args.version}/models/{prefix_name}/{round}/{batch_size}_{lr}"
    ):
        os.makedirs(f"{args.version}/models/{prefix_name}/{round}/{batch_size}_{lr}")

    # 设置设备为 CUDA 或 CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda}")
        print("use gpu")
    else:
        device = torch.device("cpu")
        print("use cpu")

    # 加载训练和验证数据集
    train_dataset = pallmediaMyDataset(
        imgpath, samplepath, round, train="trainvalidation"
    )
    validation_dataset = pallmediaMyDataset(imgpath, samplepath, round, train="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=False,
        collate_fn=_collate_fn,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=False,
        collate_fn=_collate_fn,
    )
    # 准备模型和优化器
    testmodel = mrnet
    mrnet = nn.DataParallel(mrnet)  # 使用数据并行处理

    optimizer = optim.Adam(mrnet.parameters(), lr=lr)
    scheler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)  # 学习率调度器

    # 初始化最佳验证损失和准确率
    best_val_loss = float("inf")
    best_val_acc = float(0)
    best_val_auc = float(0)

    num_epochs = args.epochs  # 总的训练轮次
    iteration_change = 0

    # 训练开始时间
    t_start_training = time.time()
    # 初始化存储训练和验证过程中的损失、准确率和AUC
    trainloss, validationloss, trainacc, validationacc, trainauc, validationauc = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    bestepoch = 0
    e = 0
    for epoch in range(num_epochs):  # 对每个训练轮次进行迭代
        e += 1
        current_lr = get_lr(optimizer)  # 获取当前学习率
        # if e>10:
        #     current_lr=lr/10

        t_start = time.time()

        # m1 = mrnet
        # m2=mrnet
        # m2=nn.DataParallel(m2)
        # file = os.listdir(f'{args.version}/models/{prefix_name}/{round}/{args.batch_size}_{args.lr}')
        # m1.load_state_dict(
        #     torch.load(f'{args.version}/models/{prefix_name}/{round}/{args.batch_size}_{args.lr}/{file[0]}'))
        # m1 = nn.DataParallel(m1)

        # 训练模型并获取训练损失和准确率等指标
        train_loss, train_auc, train_acc, trpre, trprobas, trtrue = train_model(
            mrnet,
            train_loader,
            epoch,
            num_epochs,
            optimizer,
            current_lr,
            args.log_every,
            device,
        )
        # 验证模型并获取验证损失和准确率等指标
        val_loss, val_auc, val_acc, pre, probas, y_trues, _ = evaluate_model(
            mrnet, validation_loader, device
        )
        # scheler.step()
        # 更新训练和验证指标的记录
        trainloss.append(train_loss)
        trainacc.append(train_acc)
        trainauc.append(train_auc)
        validationloss.append(val_loss)
        validationacc.append(val_acc)
        validationauc.append(val_auc)

        t_end = time.time()
        delta = t_end - t_start

        # 打印本轮的训练和验证结果
        print(
            "Epoch:{} |train loss:{} |val loss{} |train acc{} |val acc{} |elapsed time {}s".format(
                epoch + 1, train_loss, val_loss, train_acc, val_acc, np.round(delta, 2)
            )
        )

        iteration_change += 1
        # print('-' * 30)

        # 如果当前验证准确率是最佳的，更新最佳准确率和模型
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            bestepoch = epoch + 1
            testmodel = mrnet
            testpre = pre

        # 每10轮保存模型和结果
        if e % 10 == 0:
            file_name = f"{prefix_name}/{round}/{batch_size}_{lr}/epoch{e}_trainacc{train_acc}_valacc{val_acc}.pth"
            torch.save(
                mrnet.module.cpu().state_dict(), f"./{args.version}/models/{file_name}"
            )
            allmediaresult.printtxt(
                args,
                prefix_name,
                round,
                pre,
                probas,
                y_trues,
                "test_epoch"
                + str(e)
                + "_trainacc"
                + str(train_acc)
                + "_valacc"
                + str(val_acc),
                batch_size,
                lr,
            )
            allmediaresult.printtxt(
                args,
                prefix_name,
                round,
                trpre,
                trprobas,
                trtrue,
                "trainvalidation_epoch"
                + str(e)
                + "_trainacc"
                + str(train_acc)
                + "_valacc"
                + str(val_acc),
                batch_size,
                lr,
            )

         # 如果验证损失是最佳的，更新最佳验证损失
        if val_loss < best_val_loss or epoch < 10:
            # bestepoch = epoch + 1
            # testmodel=mrnet
            best_val_loss = val_loss
            iteration_change = 0
            # file_name = f'{prefix_name}/{round}/{args.batch_size}_{args.lr}/epoch_{bestepoch}.pth'
            # torch.save(mrnet.module.cpu().state_dict(), f'./{args.version}/models/{file_name}')
        gc.collect()
        torch.cuda.empty_cache()
        # if iteration_change == args.patience:
        #     break

    print("best Acc:{}".format(best_val_acc))
    # test_dataset = pallmediaMyDataset(args, imgpath, samplepath, round, train='test')
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
    #                                           drop_last=False,pin_memory=False)
    # test_loss, test_auc, test_acc, testpre, testprobas, test_trues,_ = evaluate_model(
    #     testmodel, validation_loader, device)
    # test_acc = metrics.accuracy_score(y_trues, testpre)
    print(f"final_acc:{val_acc}")
    models = os.listdir(
        f"./{args.version}/models/{prefix_name}/{round}/{batch_size}_{lr}"
    )
    # for mod in models:
    #     if mod.split('.')[0]!='epoch_'+str(bestepoch):
    #         os.remove(f'./{args.version}/models/{prefix_name}/{round}/{args.batch_size}_{args.lr}/{mod}')

    # allmediaresult.printtxt(args, prefix_name, round, testpre, testprobas, test_trues, 'test')
    allmediaresult.printtxt(
        args, prefix_name, round, pre, probas, y_trues, "final", batch_size, lr
    )
    allmediaresult.visual(
        args, prefix_name, round, e, trainloss, validationloss, "Loss", batch_size, lr
    )
    allmediaresult.visual(
        args, prefix_name, round, e, trainacc, validationacc, "Acc", batch_size, lr
    )
    allmediaresult.visual(
        args, prefix_name, round, e, trainauc, validationauc, "Auc", batch_size, lr
    )
    allmediaresult.visual(
        args,
        prefix_name,
        round,
        bestepoch,
        trainloss[0:bestepoch],
        validationloss[0:bestepoch],
        "bestLoss",
        batch_size,
        lr,
    )
    allmediaresult.visual(
        args,
        prefix_name,
        round,
        bestepoch,
        trainacc[0:bestepoch],
        validationacc[0:bestepoch],
        "bestAcc",
        batch_size,
        lr,
    )
    allmediaresult.visual(
        args,
        prefix_name,
        round,
        bestepoch,
        trainauc[0:bestepoch],
        validationauc[0:bestepoch],
        "bestAuc",
        batch_size,
        lr,
    )

    t_end_training = time.time()
    print(f"training took {t_end_training - t_start_training} s")
    return val_acc


class parameter:
    def __init__(self):
        self.lr_scheduler = "plateau"
        self.gamma = 0.5
        self.epochs = 40
        self.lr = 1e-5
        self.batch_size = 32
        self.flush_history = 0
        self.patience = 5
        self.save_model = 1
        self.log_every = 100
        # self.version = "v20240229"
        self.version = 'v20230316'
        self.version = 'v20230314'

def allmedia(args, prefix_name, imgpath, samplepath, cuda, net0):
    paramas = []
    testacc = []
    finalacc = []

    for i in range(0, 5):
        print(f"  round{i}")
        net = net0
        acc = allmediarun(args, prefix_name, i, cuda, net, imgpath, samplepath)
        paramas.append([args.batch_size, args.lr])
        finalacc.append("final")
        testacc.append("test")
        gc.collect()
        torch.cuda.empty_cache()
    return paramas, finalacc, testacc


torch.backends.cudnn.benchmark = True  # 加速
args = parameter()
setup_seed(3407)
sample = "../../sample/20220817wos4"
imgpath = "../../DataProcessing/gazepointimg/fsjoint1d12360"
# prefix_name = "conv2dconv1d_rconv1d1024(321)_2048(521)_17wos4_BN_3fc_rgb_d0.5_shuffle_"
prefix_name = '14gzresnet_17wos4_3fc_rgb_d0.5_shuffle_'
# prefix_name = "conv2dconv1d_rconv1d1024(321)_2048(521)_17wos4_BN_3fc_rgb_d0.5_shuffle_"  #原来是"conv2dconv1d_rconv1d1024(321)_2048(521)_17wos4_BN_3fc_rgb_d0.5_shuffle_"
print(args.version)
print(prefix_name)
paramas, testacc = [], []
lrs = [1e-6]
batchs = [32] #原来是32

I = [0, 1, 2, 3, 4]
ws = 13
es = [100] #原来是40
# conv12=[[1024,1024],[512,1024],[512,512],[256,512],[128,512]]
# for conv in conv12:
#     prefix_name = f'conv2dconv1d_60_rconv1d{conv[0]}(321)_{conv[1]}(521)_17wos4_BN_3fc_rgb_d0.5_shuffle_'
for lr in lrs:
    for batch in batchs:
        # acc = 0
        # for i in I:
        #     print(i)
        #     print(lr, batch)
        #     # net = model.conv2dconv1drpadding(conv[0],conv[1],[2,2,2,2])
        #     net = model.conv2dconv1drpadding(
        #         1024, 2048, [2, 2, 2, 2]
        #     )  # 原来model.conv2dconv1drpadding(1024, 2048, [2, 2, 2, 2])
        #     # net.load_state_dict(torch.load('v20230314/models/14gzresnet_conv1d1024(321)_2048(521)_512(511)_17wos4_BN_3fc_rgb_d0.5_shuffle_/0/32_1e-06/epoch1_trainacc0.5292_valacc0.5358.pth'))
        #     acc += allmediarun(args, prefix_name, i, 1, net, imgpath, sample, batch, lr)
        #     paramas.append([batch, lr])
        #     testacc.append("test")
        # acc = acc / 5
        # print(f"CV Acc: {np.round(acc,4)}")
        # # allmediaresult.vote(args,prefix_name,batch,lr,sample)
        # print(prefix_name)
        # # allmediaresult.printall(args,prefix_name, paramas, testacc,'test',sample,imgpath)
        # gc.collect()
        # torch.cuda.empty_cache()
        for e in es:
            print(e)
            allmediaresult.hardvote(args, prefix_name, batch, lr, sample, ws, e)
            allmediaresult.softvote(args, prefix_name, batch, lr, sample, ws, e)
            allmediaresult.bestsoftvote(args, prefix_name, batch, lr, sample, ws, e)
            allmediaresult.mlvote(args, prefix_name, batch, lr, sample, ws, e)

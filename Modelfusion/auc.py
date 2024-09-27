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


from dataloader import pallmediaMyDataset, _collate_fn
import model
import allmediaresult

from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")


def train_model(
    model, train_loader, epoch, num_epochs, optimizer, current_lr, log_every, device
):

    '''
    输入：
        model:模型
        train_loader:训练集
        epoch:当前epoch
        num_epochs:总epoch
        optimizer:优化器
        current_lr:当前学习率
        log_every:打印间隔
        device:设备
    输出：
        train_loss_epoch:训练集loss
        train_auc_epoch:训练集auc
        train_acc_epoch:训练集acc
    '''

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

    for i, (image1, image2, image3, label, bz) in enumerate(train_loader):
        optimizer.zero_grad()
        image1 = image1.cuda()
        image2 = image2.cuda()
        image3 = image3.cuda()
        label = label.cuda()
        # weight=weight.cuda()

        # prediction = model.forward(image1.float(),image2.float(),_)
        prediction = model.forward(
            image1.float(), image2.float(), image3.float(), bz, device
        )
        # prediction=prediction.cuda()
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weight[0])
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn=loss_fn.cuda()
        # lossdvzs=loss_fn(dvzspre,label)
        # losszs=loss_fn(zspre,label)
        # lossl=loss_fn(lpre,label)
        loss1 = loss_fn(prediction, label)
        # loss=0.5*loss1+0.3*lossl+0.2*lossdvzs
        loss = loss1
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

        for i, (image1, image2, image3, label, bz) in enumerate(val_loader):
            image1 = image1.cuda()
            image2 = image2.cuda()
            image3 = image3.cuda()
            label = label.cuda()
            # weight=weight.cuda()

            # prediction = model.forward(image1.float(),image2.float(),_)
            prediction = model.forward(
                image1.float(), image2.float(), image3.float(), bz, device
            )
            # prediction=prediction.cuda()
            out.append(prediction)

            # loss_fn = torch.nn.CrossEntropyLoss(weight=weight[0])
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn=loss_fn.cuda()
            # lossdvzs = loss_fn(dvpre, label)
            # losszs = loss_fn(zspre, label)
            # lossl=loss_fn(lpre,label)
            loss1 = loss_fn(prediction, label)
            loss = loss1
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
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def allmediarun(
    args,
    prefix_name,
    round,
    cuda,
    mrnet,
    dvimgpath,
    zsimgpath,
    limgpath,
    islong,
    padding,
    samplepath,
    batch_size,
    lr,
):
    if not os.path.exists(
        f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}"
    ):
        os.makedirs(f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}")
    else:
        files = os.listdir(
            f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}"
        )
        for file in files:
            os.remove(
                f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}/{file}"
            )
    if not os.path.exists(
        f"{args.version}/models/{prefix_name}/{round}/{batch_size}_{lr}"
    ):
        os.makedirs(f"{args.version}/models/{prefix_name}/{round}/{batch_size}_{lr}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda}")
        print("use gpu")
    else:
        device = torch.device("cpu")
        print("use cpu")

    train_dataset = pallmediaMyDataset(
        args,
        dvimgpath,
        zsimgpath,
        limgpath,
        islong,
        samplepath,
        round,
        train="trainvalidation",
        padding=padding,
    )
    validation_dataset = pallmediaMyDataset(
        args,
        dvimgpath,
        zsimgpath,
        limgpath,
        islong,
        samplepath,
        round,
        train="test",
        padding=padding,
    )
    # test_dataset=pallmediaMyDataset(args,imgpath,samplepath, round,train='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=False,
        pin_memory=False,
        collate_fn=_collate_fn,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=False,
        collate_fn=_collate_fn,
    )

    testmodel = mrnet

    mrnet = nn.DataParallel(mrnet)

    optimizer = optim.Adam(mrnet.parameters(), lr=lr)
    # scheler=optim.lr_scheduler.StepLR(optimizer,30,gamma=0.1)
    scheler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

    best_val_loss = float("inf")
    best_val_acc = float(0)
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change = 0

    t_start_training = time.time()
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
    cbatch_size = batch_size
    for epoch in range(num_epochs):
        e += 1
        # print(current_lr)

        t_start = time.time()
        # if epoch==51:
        #     current_lr=8e-8
        #     set_learning_rate(optimizer,current_lr)
        #     cbatch_size=32
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=cbatch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False)
        #     validation_loader = torch.utils.data.DataLoader(
        #         validation_dataset, batch_size=cbatch_size, shuffle=False, num_workers=16, drop_last=False,
        #         pin_memory=False)
        # elif epoch==76:
        #     cbatch_size=8
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=cbatch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False)
        #     validation_loader = torch.utils.data.DataLoader(
        #         validation_dataset, batch_size=cbatch_size, shuffle=False, num_workers=16, drop_last=False,
        #         pin_memory=False)
        # elif epoch==126:
        #     cbatch_size=4
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=cbatch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False)
        #     validation_loader = torch.utils.data.DataLoader(
        #         validation_dataset, batch_size=cbatch_size, shuffle=False, num_workers=16, drop_last=False,
        #         pin_memory=False)
        # elif epoch==176:
        #     cbatch_size=1
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=cbatch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False)
        #     validation_loader = torch.utils.data.DataLoader(
        #         validation_dataset, batch_size=cbatch_size, shuffle=False, num_workers=16, drop_last=False,
        #         pin_memory=False)
        # elif epoch==226:
        #     current_lr=8e-9
        #     set_learning_rate(optimizer,current_lr)

        # elif epoch>=160 and epoch<200:
        #     current_lr=1e-8
        # elif epoch>=200:
        #     current_lr=1e-9

        # m1 = mrnet
        # m2=mrnet
        # m2=nn.DataParallel(m2)
        # file = os.listdir(f'{args.version}/models/{prefix_name}/{round}/{args.batch_size}_{args.lr}')
        # m1.load_state_dict(
        #     torch.load(f'{args.version}/models/{prefix_name}/{round}/{args.batch_size}_{args.lr}/{file[0]}'))
        # m1 = nn.DataParallel(m1)
        # if e==50:
        #     set_learning_rate(optimizer,1e-7)
        # if e==80:
        #     set_learning_rate(optimizer,1e-8)
        current_lr = get_lr(optimizer)
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
        val_loss, val_auc, val_acc, pre, probas, y_trues, _ = evaluate_model(
            mrnet, validation_loader, device
        )
        # scheler.step()
        trainloss.append(train_loss)
        trainacc.append(train_acc)
        trainauc.append(train_auc)
        validationloss.append(val_loss)
        validationacc.append(val_acc)
        validationauc.append(val_auc)

        t_end = time.time()
        delta = t_end - t_start

        print(
            "Epoch:{}|train acc{}|val acc{}|train auc{}|val auc{}|elapsed time {}s|batch{}|lr{}".format(
                epoch + 1,
                train_acc,
                val_acc,
                train_auc,
                val_auc,
                np.round(delta, 2),
                cbatch_size,
                current_lr,
            )
        )

        iteration_change += 1
        # print('-' * 30)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            bestepoch = epoch + 1
            testmodel = mrnet
            testpre = pre
        if e % 10 == 0:
            file_name = f"{prefix_name}/{round}/{batch_size}_{lr}/epoch{e}_trainacc{train_acc}_valacc{val_acc}_trainauc{train_auc}_valauc{val_auc}.pth"
            torch.save(
                mrnet.module.cpu().state_dict(), f"./{args.version}/models/{file_name}"
            )
            print("#################################")
            print(file_name)
            # torch.save(mrnet.state_dict(), f'./{args.version}/models/{file_name}')
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
                + str(val_acc)
                + "_trainauc"
                + str(train_auc)
                + "_valauc"
                + str(val_auc),
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
                + str(val_acc)
                + "_trainauc"
                + str(train_auc)
                + "_valauc"
                + str(val_auc),
                batch_size,
                current_lr,
            )

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
    #     testmodel, test_loader, device)
    # test_acc = metrics.accuracy_score(test_trues, testpre)
    print(f"final_acc:{val_acc}")
    models = os.listdir(
        f"./{args.version}/models/{prefix_name}/{round}/{batch_size}_{lr}"
    )
    # for mod in models:
    #     if mod.split('.')[0]!='epoch_'+str(bestepoch):
    #         os.remove(f'./{args.version}/models/{prefix_name}/{round}/{args.batch_size}_{args.lr}/{mod}')

    # allmediaresult.printtxt(args, prefix_name, round, testpre, testprobas, test_trues, 'test')
    # allmediaresult.printtxt(args,prefix_name, round, pre, probas, y_trues, 'final')
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
    trainacc = np.array(trainacc)
    trainauc = np.array(trainauc)
    trainloss = np.array(trainloss)
    validationacc = np.array(validationacc)
    validationauc = np.array(validationauc)
    validationloss = np.array(validationloss)
    return (
        best_val_acc,
        trainacc,
        trainloss,
        trainauc,
        validationacc,
        validationloss,
        validationauc,
        val_acc,
    )


class parameter:
    def __init__(self):
        self.lr_scheduler = "plateau"
        self.gamma = 0.5
        self.epochs = 100
        self.lr = 1e-6
        self.batch_size = 64
        self.flush_history = 0
        self.patience = 5
        self.save_model = 1
        self.log_every = 100
        # self.version = "v20240402/dv_zssa_l"
        self.version = "v20230320/dv_zssa"    
        # self.version = "v20240316/dv_zssa_l"    



torch.backends.cudnn.benchmark = True
args = parameter()
setup_seed(3407)

"""
sample = '../../../new_data/ASDeyetracking/sample/20220817wos4'
dvimgpath = '../../../new_data/ASDeyetracking/DataProcessing/DynamicalVisualize/imgwos4_4'
zsimgpath = '../../../new_data/ASDeyetracking/DataProcessing/saliencyaddgazemmwos4'
limgpath = '../../../new_data/ASDeyetracking/DataProcessing/gazepointimg/fsjoint1d12350'
"""


sample = "../sample/20220817wos4"
dvimgpath = "../DataProcessing/DynamicalVisualize/imgwos4_4"
# zsimgpath= '../DataProcessing/gazepointimg/gazepointimgmmwos415'
zsimgpath = "../DataProcessing/saliencyaddgazemmwos4"
limgpath = "../DataProcessing/gazepointimg/fsjoint1d12350"

fcd4 = [[1, None]]
fway = "con4"

prefix_name = f"l_5fc_padding_d0.5_convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle"  # 30
dvmodelpath = "../globalheatmapclassifier/v20230223/dvmm/models/14resnet_3fc_rgb17wos4_4/d0.5_n0.5_shuffle"  # 动力学
zsmodelpath = "../globalheatmapclassifier/v20220819/saliencyaddgazemmwos4/models/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle"  # 注意力
lmodelpath = "../exp20220616/longimg/v20230316/models/conv2dconv1d_rconv1d1024(321)_2048(521)_17wos4_BN_3fc_rgb_d0.5_shuffle_"  # 时序
dvzs433modelpath = "v20230226/dv_zssa/models/convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle"


lb34 = [[64, 1e-6]]
fcd3 = [5]
es = [20]
ws = 12

I = [0, 1, 2, 3,4]
for lb in lb34:
    acc = 0
    for d in fcd3:
        # prefix_name = f'l_{d}fc_padding_d0.5_convcon_conv512_1*1_conv256_3*3_noupdate_1loss_3fc/d0.5_n0.5_shuffle'

        prefix_name = f"convcon_conv512_1*1_conv512_3*3_noupdate_1loss_3fc/d0.5_n0.5_shuffle"
        print(prefix_name)
        # prefix_name = 'convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle'

        # for i in I:
            
        #     print(i)
        #     print(lb)
        #     # net = ResNet([2,2,2,2])
        #     dvallpath=f'{dvmodelpath}/{i}/64_1e-06'
        #     dvfiles=os.listdir(dvallpath)
        #     print(dvfiles[5:8])
        #     for dvfile in dvfiles:
        #         if dvfile[5:8]==str(100):  #改了。原来是dvfile[5:7]==str(80)
        #             dvpath=f'{dvmodelpath}/{i}/64_1e-06/{dvfile}'
        #             #print("#################### YES ########################")
        #             break
        #     # zsallpath = f'{zsmodelpath}/{i}/64_1e-06'
        #     # zsfiles = os.listdir(zsallpath)
        #     # for zsfile in zsfiles:
        #     #     if zsfile[5:8] == str(100):
        #     #         zspath = f'{zsmodelpath}/{i}/64_1e-06/{zsfile}'
        #     #         break
        #     zsallpath = f'{zsmodelpath}/{i}/64_1e-06'
        #     zsfiles = os.listdir(zsallpath)
        #     for zsfile in zsfiles:
        #         if zsfile[5:7] == str(80):  #之前是zsfile[5:7] == str(80):
        #             zspath = f'{zsmodelpath}/{i}/64_1e-06/{zsfile}'
        #             break
        #     # lallpath = f'{lmodelpath}/{i}/16_6e-07'
        #     lallpath = f'{lmodelpath}/{i}/32_1e-06'
        #     lfiles = os.listdir(lallpath)
        #     for lfile in lfiles:
        #         if lfile[5:7] == str(40):
        #             lpath = f'{lmodelpath}/{i}/32_1e-06/{lfile}'
        #             break
        
        #     dvzsallpath = f'{dvzs433modelpath}/{i}/64_1e-06'
        #     dvzsfiles = os.listdir(dvzsallpath)
        #     for dvzsfile in dvzsfiles:
        #         # if dvzsfile[5:7] == str(20):
        #         if dvzsfile[5:7] == str(20): #之前是40
        #             dvzspath = f'{dvzs433modelpath}/{i}/64_1e-06/{dvzsfile}'
        #             break
        #     print(dvpath,zspath,lpath,dvzspath)
        #     net=model.dvzsfusionconv_lf_fccon(dvpath,zspath,lpath=lpath,dvzspath=dvzspath,update=True,fc=d,d=0.5,padding=1)
        #     #net = model.dvzsfusionconvcon_con(dvpath, zspath, None, False,0.5)
        #     print("****************1")
        #     _,tracc,trauc,trloss,valacc,valauc,valloss,acc1= allmediarun(args, prefix_name, i, 0, net, dvimgpath,zsimgpath, limgpath,True,1, sample,lb[0], lb[1])
        #     acc+=acc1
        #     gc.collect()
        #     torch.cuda.empty_cache()
        # acc = acc / 5
        # print(f'CV Acc: {np.round(acc, 4)}')
        # #allmediaresult.vote(args,prefix_name,lb[0],lb[1],sample)
        # allmediaresult.printpro(args, prefix_name, sample, 0, lb[0], lb[1])
        # print(prefix_name)
        # gc.collect()
        # torch.cuda.empty_cache()

        for e in es:
            print(e)
            allmediaresult.hardvote(args, prefix_name, lb[0], lb[1], sample, ws, e)
            allmediaresult.softvote(args, prefix_name, lb[0], lb[1], sample, ws, e)
            allmediaresult.bestsoftvote(args, prefix_name, lb[0], lb[1], sample, ws, e)
            allmediaresult.mlvote(args, prefix_name, lb[0], lb[1], sample, ws, e)

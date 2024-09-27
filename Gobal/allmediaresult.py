import os
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import xlwt
import torch
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import (
    KNeighborsClassifier,
    NeighborhoodComponentsAnalysis,
    NearestCentroid,
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sklearn
from sklearn import tree
import pandas as pd
import pickle



# 绘制loss、ACC、auc曲线并保存
def visual(args, prefix_name, round, e, trainloss, testloss, title, batch_size, lr):
    if not os.path.exists(f"{args.version}/result/{prefix_name}_{batch_size}_{lr}"):
        os.makedirs(f"{args.version}/result/{prefix_name}_{batch_size}_{lr}")
    x = [i for i in range(1, e + 1)]
    plt.figure()
    plt.axis([1, e, 0, 1])  ##（0.5，1）x轴的范围， （0,1.08）y轴的范围
    plt.xticks([])  ## 显示的x轴刻度值
    plt.yticks([i * 0.1 for i in range(0, 11)])  ## 显示y轴刻度值
    plt.plot(x, trainloss, color="b", label="Train")
    plt.plot(x, testloss, color="r", label="Validation")
    plt.grid()
    # 显示图例（使绘制生效）
    plt.legend()
    # 横坐标名称
    plt.xlabel(f"{e}epoch")
    plt.ylabel(title)
    plt.title("Round" + str(round))
    # 纵坐标名称
    # 保存图片到本地
    plt.savefig(
        f"{args.version}/result/{prefix_name}_{batch_size}_{lr}/{title}_{round}.png"
    )
    plt.close()


# 保存分类标签与概率
def printtxt(args, prefix_name, round, pre, probas, y_trues, bestacc, batch_size, lr):
    if not os.path.exists(
        f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}"
    ):
        os.makedirs(f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}")
    with open(
        f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}/{bestacc}.txt",
        "w",
    ) as f:
        for i in range(len(y_trues)):
            f.write(str(y_trues[i]))
            f.write("\t")
            f.write(str(pre[i]))
            f.write("\t")
            f.write(str(probas[i]))
            f.write("\n")


# 根据保存的结果文件，计算某一折sample级的性能指标，打印出来
def individualperfor(args, prefix_name, round, batch_size, lr, bestacc, imgpath):
    with open(f"{imgpath}/{round}/testsample.txt", "r") as fo:
        myFile = fo.read()
        myRecords = myFile.split("\n")
        subject = []
        label = []  # sample真实标签
        fsample = []
        for y in range(0, len(myRecords) - 1):
            subject.append(myRecords[y].split("\t"))  # 把表格数据存入
        fsample = subject
        for i in range(0, len(fsample)):
            if fsample[i][1][0] == "n":
                label.append(0)
            else:
                label.append(1)

    with open(
        f"{args.version}/output/{prefix_name}_{round}/{batch_size}_{lr}/{bestacc}.txt",
        "r",
    ) as fo:
        myFile = fo.read()
        myRecords = myFile.split("\n")
        subject, pre, pro = [], [], []
        for y in range(0, len(myRecords) - 1):
            subject.append(myRecords[y].split("\t"))  # 把表格数据存入
        for i in range(0, len(subject)):
            pre.append(int(subject[i][1]))  # 预测标签
            pro.append(float(subject[i][2]))  # 预测为1的概率
    auc1 = metrics.roc_auc_score(label, pro)
    accuracy1 = metrics.accuracy_score(label, pre)
    precision1 = metrics.precision_score(label, pre)
    recall1 = metrics.recall_score(label, pre)
    f11 = metrics.f1_score(label, pre)
    tn, fp, fn, tp = metrics.confusion_matrix(label, pre).ravel()
    specificity = tn / (tn + fp)
    if not os.path.exists(f"{args.version}/result/{prefix_name}"):
        os.makedirs(f"{args.version}/result/{prefix_name}")
    with open(f"{args.version}/result/{prefix_name}/{batch_size}_{lr}.txt", "a") as f:
        print(round)
        f.write(str(round))
        f.write("\n")
        print("auc:" + str(auc1))
        f.write("auc:" + str(auc1))
        f.write("\n")
        print("accuracy:" + str(accuracy1))
        f.write("accuracy:" + str(accuracy1))
        f.write("\n")
        print("precision:" + str(precision1))
        f.write("precision:" + str(precision1))
        f.write("\n")
        print("recall:" + str(recall1))
        f.write("recall:" + str(recall1))
        f.write("\n")
        print("f1:" + str(f11))
        f.write("f1:" + str(f11))
        f.write("\n")
        print("specificity:" + str(specificity))
        f.write("specificity:" + str(specificity))
        f.write("\n")
    return [auc1, accuracy1, precision1, recall1, f11, specificity]


# 计算交叉验证结果指标，打印
def meanperfor(args, prefix_name, params, bestacc, imgpath):
    estimate = []
    estimate.append(
        individualperfor(
            args, prefix_name, 0, params[0][0], params[0][1], bestacc[0], imgpath
        )
    )
    estimate.append(
        individualperfor(
            args, prefix_name, 1, params[1][0], params[1][1], bestacc[1], imgpath
        )
    )
    estimate.append(
        individualperfor(
            args, prefix_name, 2, params[2][0], params[2][1], bestacc[2], imgpath
        )
    )
    estimate.append(
        individualperfor(
            args, prefix_name, 3, params[3][0], params[3][1], bestacc[3], imgpath
        )
    )
    estimate.append(
        individualperfor(
            args, prefix_name, 4, params[4][0], params[4][1], bestacc[4], imgpath
        )
    )
    reauc, reacc, repre, rere, ref1, resp = [], [], [], [], [], []
    for i in range(len(estimate)):
        reauc.append(estimate[i][0])
        reacc.append(estimate[i][1])
        repre.append(estimate[i][2])
        rere.append(estimate[i][3])
        ref1.append(estimate[i][4])
        resp.append(estimate[i][5])
    meanaucdv = np.mean(reauc)
    meanaccdv = np.mean(reacc)
    meanprecisiondv = np.mean(repre)
    meanrecalldv = np.mean(rere)
    meanf1dv = np.mean(ref1)
    meanspecificty = np.mean(resp)
    varaucdv = np.var(reauc)
    varaccdv = np.var(reacc)
    varprecisiondv = np.var(repre)
    varrecalldv = np.var(rere)
    varf1dv = np.var(ref1)
    varspecificty = np.var(resp)
    with open(
        f"{args.version}/result/{prefix_name}/{args.batch_size}_{args.lr}.txt", "a"
    ) as f:
        print("auc:" + str(meanaucdv) + "+/-" + str(varaucdv))
        f.write("auc:" + str(meanaucdv) + "+/-" + str(varaucdv))
        f.write("\n")
        print("accuracy:" + str(meanaccdv) + "+/-" + str(varaccdv))
        f.write("accuracy:" + str(meanaccdv) + "+/-" + str(varaccdv))
        f.write("\n")
        print("precision:" + str(meanprecisiondv) + "+/-" + str(varprecisiondv))
        f.write("precision:" + str(meanprecisiondv) + "+/-" + str(varprecisiondv))
        f.write("\n")
        print("recall:" + str(meanrecalldv) + "+/-" + str(varrecalldv))
        f.write("recall:" + str(meanrecalldv) + "+/-" + str(varrecalldv))
        f.write("\n")
        print("f1:" + str(meanf1dv) + "+/-" + str(varf1dv))
        f.write("f1:" + str(meanf1dv) + "+/-" + str(varf1dv))
        f.write("\n")
        print("specificity:" + str(meanspecificty) + "+/-" + str(varspecificty))
        f.write("specificity:" + str(meanspecificty) + "+/-" + str(varspecificty))
        f.write("\n")
    return [
        meanaccdv,
        meanprecisiondv,
        meanrecalldv,
        meanf1dv,
        meanspecificty,
        meanaucdv,
    ]


# 输入真实标签和预测标签，求性能指标
def calestimate(dtest1y, dtapre1, dtapro1):
    auc1 = metrics.roc_auc_score(dtest1y, dtapro1)
    accuracy1 = metrics.accuracy_score(dtest1y, dtapre1)
    precision1 = metrics.precision_score(dtest1y, dtapre1)
    recall1 = metrics.recall_score(dtest1y, dtapre1)
    f11 = metrics.f1_score(dtest1y, dtapre1)
    tn, fp, fn, tp = metrics.confusion_matrix(dtest1y, dtapre1).ravel()
    specificity = tn / (tn + fp)
    return [auc1, accuracy1, precision1, recall1, f11, specificity]


# 输入已经计算好的各折性能指标，计算均值和方差，并打印
def printresult(args, prefix_name, estimate):
    print("vote")
    reauc, reacc, repre, rere, ref1, resp = [], [], [], [], [], []
    for i in range(len(estimate)):
        reauc.append(estimate[i][0])
        reacc.append(estimate[i][1])
        repre.append(estimate[i][2])
        rere.append(estimate[i][3])
        ref1.append(estimate[i][4])
        resp.append(estimate[i][5])
    print(estimate)
    meanaucdv = np.mean(reauc)
    meanaucdv = np.round(meanaucdv, 4)    
    meanaccdv = np.mean(reacc)
    meanaccdv = np.round(meanaccdv, 4)
    meanprecisiondv = np.mean(repre)
    meanprecisiondv = np.round(meanprecisiondv, 4)
    meanrecalldv = np.mean(rere)
    meanrecalldv = np.round(meanrecalldv, 4)
    meanf1dv = np.mean(ref1)
    meanf1dv = np.round(meanf1dv, 4)
    meanspecificty = np.mean(resp)
    meanspecificty = np.round(meanspecificty, 4)
    varaucdv = np.var(reauc)
    varaucdv = np.round(varaucdv, 4)
    varaccdv = np.std(reacc)
    varaccdv = np.round(varaccdv, 4)
    varprecisiondv = np.std(repre)
    varprecisiondv = np.round(varprecisiondv, 4)
    varrecalldv = np.std(rere)
    varrecalldv = np.round(varrecalldv, 4)
    varf1dv = np.std(ref1)
    varf1dv = np.round(varf1dv, 4)
    varspecificty = np.std(resp)
    varspecificty = np.round(varspecificty, 4)
    # with open(f'{args.version}/result/{prefix_name}/{args.batch_size}_{args.lr}.txt','a') as f:
    print('auc:' + str(meanaucdv) + '+/-' + str(varaucdv))
    # f.write('auc:' + str(meanaucdv) + '+/-' + str(varaucdv))
    # f.write('\n')
    print("accuracy:" + str(meanaccdv) + "+/-" + str(varaccdv))
    # f.write('accuracy:' + str(meanaccdv) + '+/-' + str(varaccdv))
    # f.write('\n')
    print("precision:" + str(meanprecisiondv) + "+/-" + str(varprecisiondv))
    # f.write('precision:' + str(meanprecisiondv) + '+/-' + str(varprecisiondv))
    # f.write('\n')
    print("recall:" + str(meanrecalldv) + "+/-" + str(varrecalldv))
    # f.write('recall:' + str(meanrecalldv) + '+/-' + str(varrecalldv))
    # f.write('\n')
    print("f1:" + str(meanf1dv) + "+/-" + str(varf1dv))
    # f.write('f1:' + str(meanf1dv) + '+/-' + str(varf1dv))
    # f.write('\n')
    print("specificity:" + str(meanspecificty) + "+/-" + str(varspecificty))
    # f.write('specificity:' + str(meanspecificty) + '+/-' + str(varspecificty))
    # f.write('\n')
    return meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty


def hardvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
    print("hard vote")
    staestimatesub = []
    videoesti = []
    for i in range(5):
        allpre = []
        allpro = []
        real = []
        plabel = []
        subject_pro = []
        with open(f"{sample}/participant14.txt", "r") as f:
            myfile = f.read()
            records = myfile.split("\n")
            participants = records[i].split("\t")
            for j in range(len(participants) - 1):
                if participants[j] == "":
                    pass
                elif participants[j][0] == "n":  # 新数据集是t,旧数据集是n
                    real.append(0)
                else:
                    real.append(1)

        mediasample = []
        mep = []
        fmep = []
        with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
            mef = f.read()
            mer = mef.split("\n")
            for x in range(len(mer) - 1):
                mep.append(mer[x].split("\t"))

        mediasample = mep
        files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
        for file in files:
            if file[:ws] == "test_epoch" + str(bestepoch):
                name = file
                break
        with open(
            f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
        ) as fo:
            myFile = fo.read()
            myRecords = myFile.split("\n")
            subject, pre, pro, videolabel = [], [], [], []
            for y in range(0, len(myRecords) - 1):
                subject.append(myRecords[y].split("\t"))  # 把表格数据存入
            for z in range(0, len(subject)):
                pre.append(int(subject[z][1]))
                pro.append(float(subject[z][2]))
                videolabel.append(int(subject[z][0]))
            allpre.extend(pre)
            allpro.extend(pro)
        videoesti.append(calestimate(videolabel, pre,pro))

        for j in range(len(real)):
            reord = []
            for y in range(len(mediasample)):
                if mediasample[y][1][: len(participants[j])] == participants[j]:
                    # ###分数低的不参与投票
                    # if allpro[y]>=0.6 or allpro[y]<=0.4:
                    reord.append(int(allpre[y]))
            subject_pro.append(sum(reord) / len(reord))
            if reord.count(1) >= reord.count(0):
                plabel.append(1)
            else:
                plabel.append(0)
        staestimatesub.append(calestimate(real, plabel,subject_pro))

    _ = printresult(args, prefix_name, videoesti)
    meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
        args, prefix_name, staestimatesub
    )
    return ["vote", meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty]


def softvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
    print("soft vote")
    staestimatesub = []
    videoesti = []
    for i in range(5):
        allpre = []
        allpro = []
        real = []
        plabel = []
        subject_pro = []
        with open(f"{sample}/participant14.txt", "r") as f:
            myfile = f.read()
            records = myfile.split("\n")
            participants = records[i].split("\t")
            for j in range(len(participants) - 1):
                if participants[j] == "":
                    pass
                elif participants[j][0] == "n":
                    real.append(0)
                else:
                    real.append(1)

        mediasample = []
        mep = []
        fmep = []
        with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
            mef = f.read()
            mer = mef.split("\n")
            for x in range(len(mer) - 1):
                mep.append(mer[x].split("\t"))
        mediasample = mep
        files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
        for file in files:
            if file[:ws] == "test_epoch" + str(bestepoch):
                name = file
                break
        with open(
            f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
        ) as fo:
            myFile = fo.read()
            myRecords = myFile.split("\n")
            subject, pre, pro, videolabel = [], [], [], []
            for y in range(0, len(myRecords) - 1):
                subject.append(myRecords[y].split("\t"))  # 把表格数据存入
            for z in range(0, len(subject)):
                pre.append(int(subject[z][1]))
                pro.append(float(subject[z][2]))
                videolabel.append(int(subject[z][0]))
            allpre.extend(pre)
            allpro.extend(pro)
        videoesti.append(calestimate(videolabel, pre,pro))

        for j in range(len(real)):
            reord = []
            pro0, pro1 = 0, 0
            for y in range(len(mediasample)):
                if mediasample[y][1][: len(participants[j])] == participants[j]:
                    reord.append(int(allpre[y]))
                    pro1 += allpro[y]
            
            if pro1 > len(reord) - pro1:
                plabel.append(1)
            else:
                plabel.append(0)
            pro1 /= len(reord)
            subject_pro.append(pro1)

        staestimatesub.append(calestimate(real, plabel,subject_pro))
    meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
        args, prefix_name, staestimatesub
    )
    return ["vote", meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty]


def bestsoftvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
    print("bestsoft vote")
    staestimatesub = []
    videoesti = []
    bestacc = 0
    thre = 0
    for t in range(1, 100):
        softstaestimatesub = []
        for i in range(5):
            allpre = []
            allpro = []
            real = []
            plabel = []
            softplabel = []
            subject_pro = []
            with open(f"{sample}/participant14.txt", "r") as f:
                myfile = f.read()
                records = myfile.split("\n")
                participants = records[i].split("\t")
                for j in range(len(participants) - 1):
                    if participants[j] == "":
                        pass
                    elif participants[j][0] == "n":
                        real.append(0)
                    else:
                        real.append(1)

            mediasample = []
            mep = []
            fmep = []
            with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
                mef = f.read()
                mer = mef.split("\n")
                for x in range(len(mer) - 1):
                    mep.append(mer[x].split("\t"))
            mediasample = mep
            files = os.listdir(
                f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}"
            )
            for file in files:
                if file[:ws] == "test_epoch" + str(bestepoch):
                    name = file
                    break
            with open(
                f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
            ) as fo:
                myFile = fo.read()
                myRecords = myFile.split("\n")
                subject, pre, pro, videolabel = [], [], [], []
                for y in range(0, len(myRecords) - 1):
                    subject.append(myRecords[y].split("\t"))  # 把表格数据存入
                for z in range(0, len(subject)):
                    pre.append(int(subject[z][1]))
                    pro.append(float(subject[z][2]))
                    videolabel.append(int(subject[z][0]))
                allpre.extend(pre)
                allpro.extend(pro)
            videoesti.append(calestimate(videolabel, pre,pro))

            for j in range(len(real)):
                reord = []
                pro0, pro1 = 0, 0
                for y in range(len(mediasample)):
                    if mediasample[y][1][: len(participants[j])] == participants[j]:
                        pro1 += allpro[y]
                        reord.append(int(allpre[y]))
                pro1 /= len(reord)
                if pro1 >= 0.01 * t:
                    softplabel.append(1)
                else:
                    softplabel.append(0)
                subject_pro.append(pro1)

            softstaestimatesub.append(calestimate(real, softplabel,subject_pro))
        res = [re[1] for re in softstaestimatesub]
        acc = np.mean(res)
        if acc > bestacc:
            bestacc = acc
            thre = t
            staestimatesub = softstaestimatesub
    print(thre)
    meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
        args, prefix_name, staestimatesub
    )
    # _=printresult(args,prefix_name,videoesti)
    return ["vote", meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty]


# def bestvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
#     print("best result")
#     staestimatesub = []
#     videoesti = []
#     with open(f"{sample}/participant14.txt", "r") as f:
#         myfile = f.read()
#         records = myfile.split("\n")
#     for i in range(5):
#         allpre = []
#         allpro1, allpro0 = [], []
#         real = []
#         plabel = []
#         participants = records[i].split("\t")
#         for j in range(len(participants) - 1):
#             if participants[j] == "":
#                 pass
#             elif participants[j][0] == "n":
#                 real.append(0)
#             else:
#                 real.append(1)

#         mediasample = []
#         mep = []
#         fmep = []
#         with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
#             mef = f.read()
#             mer = mef.split("\n")
#             for x in range(len(mer) - 1):
#                 mep.append(mer[x].split("\t"))
#         mediasample = mep
#         files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
#         for file in files:
#             if file[:ws] == "test_epoch" + str(bestepoch):
#                 name = file
#                 break
#         with open(
#             f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
#         ) as fo:
#             myFile = fo.read()
#             myRecords = myFile.split("\n")
#             subject, pre, pro1, pro0, videolabel = [], [], [], [], []
#             for y in range(0, len(myRecords) - 1):
#                 subject.append(myRecords[y].split("\t"))  # 把表格数据存入
#             for z in range(0, len(subject)):
#                 pre.append(int(subject[z][1]))
#                 pro1.append(float(subject[z][2]))
#                 pro0.append(1 - float(subject[z][2]))
#                 videolabel.append(int(subject[z][0]))
#             allpre = pre
#             allpro1 = pro1
#             allpro0 = pro0
#         videoesti.append(calestimate(videolabel, pre))

#         for j in range(len(real)):
#             record1, record0 = [], []
#             pro0, pro1 = 0, 0
#             for y in range(len(mediasample)):
#                 if mediasample[y][1][: len(participants[j])] == participants[j]:
#                     record1.append(allpro1[y])
#                     record0.append(allpro0[y])
#             pro1 = max(record1)
#             pro0 = max(record0)
#             if pro1 >= pro0:
#                 plabel.append(1)
#             else:
#                 plabel.append(0)

#         staestimatesub.append(calestimate(real, plabel))
#     meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
#         args, prefix_name, staestimatesub
#     )


# def mlvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
#     print("ML result")
#     svmstaestimatesub = []
#     lstaestimatesub = []
#     knnestimatesub = []
#     dtstaestimatesub = []
#     rfstaestimatesub = []
#     videoesti = []
#     participants = []
#     reals = []
#     mediasample = []
#     allpre, allpro, records = [], [], []
#     with open(f"{sample}/participant14.txt", "r") as f:
#         myfile = f.read()
#         myrecords = myfile.split("\n")
#     for i in range(5):
#         real = []
#         participant = myrecords[i].split("\t")
#         participants.append(participant)
#         for j in range(len(participant) - 1):
#             if participant[j] == "":
#                 pass
#             elif participant[j][0] == "n":
#                 real.append(0)
#             else:
#                 real.append(1)
#         reals.append(real)
#         mep = []
#         with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
#             mef = f.read()
#             mer = mef.split("\n")
#             for x in range(len(mer) - 1):
#                 mep.append(mer[x].split("\t"))
#         mediasample.append(mep)
#         files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
#         for file in files:
#             if file[:ws] == "test_epoch" + str(bestepoch):
#                 name = file
#                 break
#         with open(
#             f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
#         ) as fo:
#             myFile = fo.read()
#             myRecords = myFile.split("\n")
#             subject, pre, pro1, pro0, videolabel = [], [], [], [], []
#             for y in range(0, len(myRecords) - 1):
#                 subject.append(myRecords[y].split("\t"))  # 把表格数据存入
#             for z in range(0, len(subject)):
#                 pre.append(int(subject[z][1]))
#                 pro1.append(float(subject[z][2]))
#                 pro0.append(1 - float(subject[z][2]))
#                 videolabel.append(int(subject[z][0]))
#             allpre.append(pre)
#             allpro.append(pro1)

#     for i in range(len(reals)):
#         record = []
#         for j in range(len(reals[i])):
#             record1 = []
#             for m in range(1, 15):
#                 for r in range(1, 4):
#                     bz = 0
#                     for y in range(len(mediasample[i])):
#                         if mediasample[i][y][1] == participants[i][j] + str(
#                             r
#                         ) and mediasample[i][y][0] == "media" + str(m):
#                             bz = 1
#                             record1.append(allpro[i][y])
#                             break
#                     if bz == 0:
#                         record1.append(0.5)
#             record.append(record1)
#         records.append(record)

#     kernel = ["rbf", "poly", "sigmoid"]
#     C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#     gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]

#     Cs = [1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

#     Nestimater = [3, 4, 5, 6, 7, 8, 9, 10]
#     weights = ["uniform", "distance"]
#     P = [1, 2, 3, 4, 5, 6]
#     algorthm = ["auto", "ball_tree", "kd_tree", "brute"]

#     criterion = ["entropy", "gini"]
#     splitter = ["best", "random"]
#     split = [2, 3, 4, 5, 6]
#     maxdep = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#     nemi = [5, 10, 25, 50, 100, 200]
#     maxd = [10, 30, 50, 70, 90]
#     maxf = [None, "sqrt", "log2"]

#     trainset, trainl, valset, vall = [], [], [], []
#     for i in range(5):
#         valset.append(records[i])
#         vall.append(reals[i])
#         tras, tral = [], []
#         for j in range(5):
#             if j != i:
#                 tras.extend(records[j])
#                 tral.extend(reals[j])
#         trainX2, trainlabel2 = sklearn.utils.shuffle(tras, tral, random_state=42)
#         trainset.append(trainX2)
#         trainl.append(trainlabel2)
#     svmacc, logacc, knnacc, dtacc, rfacc = 0, 0, 0, 0, 0
#     svmpre, svmpro, logpre, logpro, knnpre, knnpro, dtpre, dtpro, rfpre, rfpro = (
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#     )
#     svmp, logp, knnp, dtp, rfp = {}, {}, {}, {}, {}
#     for k in Nestimater:
#         for g in weights:
#             for c in P:
#                 for a in algorthm:
#                     score = 0
#                     stapre1, pro1 = [], []
#                     for r in range(5):
#                         svm = KNeighborsClassifier(
#                             n_neighbors=k, weights=g, p=c, algorithm=a
#                         )
#                         svm.fit(trainset[r], trainl[r])
#                         rpre = svm.predict(valset[r])
#                         # rpro=svm.predict_proba(valset[r])
#                         stapre1.append(rpre)
#                         # pro1.append(rpro)
#                         score += metrics.accuracy_score(rpre, vall[r])
#                     score /= 5
#                     if score > knnacc:
#                         knnacc = score
#                         knnpre = stapre1
#                         knnpro = pro1
#                         knnp = {"n_neighbors": k, "weights": g, "p": c, "algorithm": a}
#     for k in criterion:
#         for g in splitter:
#             for c in split:
#                 for a in maxdep:
#                     score = 0
#                     stapre1, pro1 = [], []
#                     for r in range(5):
#                         svm = tree.DecisionTreeClassifier(
#                             criterion=k,
#                             random_state=42,
#                             splitter=g,
#                             max_depth=a,
#                             min_samples_split=c,
#                             class_weight="balanced",
#                         )
#                         svm.fit(trainset[r], trainl[r])
#                         rpre = svm.predict(valset[r])
#                         rpro = svm.predict_proba(valset[r])
#                         stapre1.append(rpre)
#                         pro1.append(rpro)
#                         score += metrics.accuracy_score(rpre, vall[r])
#                     score /= 5
#                     if score > dtacc:
#                         dtacc = score
#                         dtpre = stapre1
#                         dtpro = pro1
#                         dtp = {
#                             "criterion": k,
#                             "splitter": g,
#                             "max_depth": a,
#                             "min_samples_split": c,
#                         }
#     for k in nemi:
#         for g in maxd:
#             for c in maxf:
#                 score = 0
#                 stapre1, pro1 = [], []
#                 for r in range(5):
#                     svm = RandomForestClassifier(
#                         n_estimators=k,
#                         max_depth=g,
#                         max_features=c,
#                         random_state=42,
#                         class_weight="balanced",
#                     )
#                     svm.fit(trainset[r], trainl[r])
#                     rpre = svm.predict(valset[r])
#                     rpro = svm.predict_proba(valset[r])
#                     stapre1.append(rpre)
#                     pro1.append(rpro)
#                     score += metrics.accuracy_score(rpre, vall[r])
#                 score /= 5
#                 if score > rfacc:
#                     rfacc = score
#                     rfpre = stapre1
#                     rfpro = pro1
#                     rfp = {"n_estimators": k, "max_depth": g, "max_features": c}
#     for c in Cs:
#         score = 0
#         stapre1, pro1 = [], []
#         for r in range(5):
#             svm = LogisticRegression(C=c, random_state=42)
#             svm.fit(trainset[r], trainl[r])
#             rpre = svm.predict(valset[r])
#             rpro = svm.predict_proba(valset[r])
#             stapre1.append(rpre)
#             pro1.append(rpro)
#             score += metrics.accuracy_score(rpre, vall[r])
#         score /= 5
#         if score > logacc:
#             logacc = score
#             logpre = stapre1
#             logpro = pro1
#             logp = {"C": c}
#     solver1 = ["svd", "lsqr", "eigen"]
#     shrink = ["auto", None]

#     for k in kernel:
#         for g in gamma:
#             for c in C:
#                 score = 0
#                 stapre1, pro1 = [], []
#                 for r in range(5):
#                     svm = SVC(
#                         C=c,
#                         kernel=k,
#                         gamma=g,
#                         probability=True,
#                         max_iter=10000,
#                         random_state=42,
#                     )
#                     svm.fit(trainset[r], trainl[r])
#                     rpre = svm.predict(valset[r])
#                     rpro = svm.predict_proba(valset[r])
#                     stapre1.append(rpre)
#                     pro1.append(rpro)
#                     score += metrics.accuracy_score(rpre, vall[r])
#                 score /= 5
#                 if score > svmacc:
#                     svmacc = score
#                     svmpre = stapre1
#                     svmpro = pro1
#                     svmp = {"gamma": g, "C": c, "kernel": k}
#     for i in range(5):
#         svmstaestimatesub.append(calestimate(vall[i], svmpre[i]))
#         lstaestimatesub.append(calestimate(vall[i], logpre[i]))
#         knnestimatesub.append(calestimate(vall[i], knnpre[i]))
#         dtstaestimatesub.append(calestimate(vall[i], dtpre[i]))
#         rfstaestimatesub.append(calestimate(vall[i], rfpre[i]))
#     print("svm")
#     print(svmp)
#     meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
#         args, prefix_name, svmstaestimatesub
#     )
#     print("logisticRegression")
#     print(logp)
#     _ = printresult(args, prefix_name, lstaestimatesub)
#     print("knn")
#     print(knnp)
#     _ = printresult(args, prefix_name, knnestimatesub)
#     print("DTree")
#     print(dtp)
#     _ = printresult(args, prefix_name, dtstaestimatesub)
#     print("RF")
#     print(rfp)
#     _ = printresult(args, prefix_name, rfstaestimatesub)


def mlvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
    print("ML result")
    svmstaestimatesub = []
    lstaestimatesub = []
    knnestimatesub = []
    dtstaestimatesub = []
    rfstaestimatesub = []
    videoesti = []
    participants = []
    reals = []
    mediasample = []
    allpre, allpro, records = [], [], []
    with open(f"{sample}/participant14.txt", "r") as f:
        myfile = f.read()
        myrecords = myfile.split("\n")
    for i in range(5):
        real = []
        participant = myrecords[i].split("\t")
        participants.append(participant)
        for j in range(len(participant) - 1):
            if participant[j] == "":
                pass
            elif participant[j][0] == "n":
                real.append(0)
            else:
                real.append(1)
        reals.append(real)
        mep = []
        with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
            mef = f.read()
            mer = mef.split("\n")
            for x in range(len(mer) - 1):
                mep.append(mer[x].split("\t"))
        mediasample.append(mep)
        files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
        for file in files:
            if file[:ws] == "test_epoch" + str(bestepoch):
                name = file
                break
        with open(
            f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
        ) as fo:
            myFile = fo.read()
            myRecords = myFile.split("\n")
            subject, pre, pro1, pro0, videolabel = [], [], [], [], []
            for y in range(0, len(myRecords) - 1):
                subject.append(myRecords[y].split("\t"))  # 把表格数据存入
            for z in range(0, len(subject)):
                pre.append(int(subject[z][1]))
                pro1.append(float(subject[z][2]))
                pro0.append(1 - float(subject[z][2]))
                videolabel.append(int(subject[z][0]))
            allpre.append(pre)
            allpro.append(pro1)

    for i in range(len(reals)):
        record = []
        for j in range(len(reals[i])):
            record1 = []
            for m in range(1, 15):
                for r in range(1, 4):
                    bz = 0
                    for y in range(len(mediasample[i])):
                        if mediasample[i][y][1] == participants[i][j] + str(
                            r
                        ) and mediasample[i][y][0] == "media" + str(m):
                            bz = 1
                            record1.append(allpro[i][y])
                            break
                    if bz == 0:
                        record1.append(0.5)
            record.append(record1)
        records.append(record)

    kernel = ["rbf", "poly", "sigmoid"]
    C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    Cs = [1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    Nestimater = [3, 4, 5, 6, 7, 8, 9, 10]
    weights = ["uniform", "distance"]
    P = [1, 2, 3, 4, 5, 6]
    algorthm = ["auto", "ball_tree", "kd_tree", "brute"]

    criterion = ["entropy", "gini"]
    splitter = ["best", "random"]
    split = [2, 3, 4, 5, 6]
    maxdep = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    nemi = [5, 10, 25, 50, 100, 200]
    maxd = [10, 30, 50, 70, 90]
    maxf = [None, "sqrt", "log2"]

    trainset, trainl, valset, vall = [], [], [], []
    for i in range(5):
        valset.append(records[i])
        vall.append(reals[i])
        tras, tral = [], []
        for j in range(5):
            if j != i:
                tras.extend(records[j])
                tral.extend(reals[j])
        trainX2, trainlabel2 = sklearn.utils.shuffle(tras, tral, random_state=42)
        trainset.append(trainX2)
        trainl.append(trainlabel2)
    svmacc, logacc, knnacc, dtacc, rfacc = 0, 0, 0, 0, 0
    svmpre, svmpro, logpre, logpro, knnpre, knnpro, dtpre, dtpro, rfpre, rfpro = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    svmp, logp, knnp, dtp, rfp = {}, {}, {}, {}, {}
    for k in Nestimater:
        for g in weights:
            for c in P:
                for a in algorthm:
                    score = 0
                    stapre1, pro1 = [], []
                    for r in range(5):
                        svm = KNeighborsClassifier(
                            n_neighbors=k, weights=g, p=c, algorithm=a
                        )
                        svm.fit(trainset[r], trainl[r])
                        rpre = svm.predict(valset[r])
                        rpro=svm.predict_proba(valset[r])
                        stapre1.append(rpre)
                        pro1.append(rpro)
                        score += metrics.accuracy_score(rpre, vall[r])
                    score /= 5
                    if score > knnacc:
                        knnacc = score
                        knnpre = stapre1
                        knnpro = pro1
                        knnp = {"n_neighbors": k, "weights": g, "p": c, "algorithm": a}
    for k in criterion:
        for g in splitter:
            for c in split:
                for a in maxdep:
                    score = 0
                    stapre1, pro1 = [], []
                    for r in range(5):
                        svm = tree.DecisionTreeClassifier(
                            criterion=k,
                            random_state=42,
                            splitter=g,
                            max_depth=a,
                            min_samples_split=c,
                            class_weight="balanced",
                        )
                        svm.fit(trainset[r], trainl[r])
                        rpre = svm.predict(valset[r])
                        rpro = svm.predict_proba(valset[r])
                        stapre1.append(rpre)
                        pro1.append(rpro)
                        score += metrics.accuracy_score(rpre, vall[r])
                    score /= 5
                    if score > dtacc:
                        dtacc = score
                        dtpre = stapre1
                        dtpro = pro1
                        dtp = {
                            "criterion": k,
                            "splitter": g,
                            "max_depth": a,
                            "min_samples_split": c,
                        }
    for k in nemi:
        for g in maxd:
            for c in maxf:
                score = 0
                stapre1, pro1 = [], []
                for r in range(5):
                    svm = RandomForestClassifier(
                        n_estimators=k,
                        max_depth=g,
                        max_features=c,
                        random_state=42,
                        class_weight="balanced",
                    )
                    svm.fit(trainset[r], trainl[r])
                    rpre = svm.predict(valset[r])
                    rpro = svm.predict_proba(valset[r])
                    stapre1.append(rpre)
                    pro1.append(rpro)
                    score += metrics.accuracy_score(rpre, vall[r])
                score /= 5
                if score > rfacc:
                    rfacc = score
                    rfpre = stapre1
                    rfpro = pro1
                    rfp = {"n_estimators": k, "max_depth": g, "max_features": c}
    for c in Cs:
        score = 0
        stapre1, pro1 = [], []
        for r in range(5):
            svm = LogisticRegression(C=c, random_state=42)
            svm.fit(trainset[r], trainl[r])
            rpre = svm.predict(valset[r])
            rpro = svm.predict_proba(valset[r])
            stapre1.append(rpre)
            pro1.append(rpro)
            score += metrics.accuracy_score(rpre, vall[r])
        score /= 5
        if score > logacc:
            logacc = score
            logpre = stapre1
            logpro = pro1
            logp = {"C": c}
    solver1 = ["svd", "lsqr", "eigen"]
    shrink = ["auto", None]

    for k in kernel:
        for g in gamma:
            for c in C:
                score = 0
                stapre1, pro1 = [], []
                for r in range(5):
                    svm = SVC(
                        C=c,
                        kernel=k,
                        gamma=g,
                        probability=True,
                        max_iter=10000,
                        random_state=42,
                    )
                    svm.fit(trainset[r], trainl[r])
                    rpre = svm.predict(valset[r])
                    rpro = svm.predict_proba(valset[r])
                    stapre1.append(rpre)
                    pro1.append(rpro)
                    score += metrics.accuracy_score(rpre, vall[r])
                score /= 5
                if score > svmacc:
                    svmacc = score
                    svmpre = stapre1
                    svmpro = pro1
                    svmp = {"gamma": g, "C": c, "kernel": k}

    for i in range(5):
        svmstaestimatesub.append(calestimate(vall[i], svmpre[i], svmpro[i][:,1]))
        lstaestimatesub.append(calestimate(vall[i], logpre[i], logpro[i][:,1]))
        knnestimatesub.append(calestimate(vall[i], knnpre[i], knnpro[i][:,1]))
        dtstaestimatesub.append(calestimate(vall[i], dtpre[i], dtpro[i][:,1]))
        rfstaestimatesub.append(calestimate(vall[i], rfpre[i], rfpro[i][:,1]))
    print("svm")
    print(svmp)
    meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
        args, prefix_name, svmstaestimatesub
    )
    print("logisticRegression")
    print(logp)
    _ = printresult(args, prefix_name, lstaestimatesub)
    print("knn")
    print(knnp)
    _ = printresult(args, prefix_name, knnestimatesub)
    print("DTree")
    print(dtp)
    _ = printresult(args, prefix_name, dtstaestimatesub)
    print("RF")
    print(rfp)
    _ = printresult(args, prefix_name, rfstaestimatesub)

    all_trainset = records[0] + records[1] + records[2] + records[3] + records[4]
    all_trainl = reals[0] + reals[1] + reals[2] + reals[3] + reals[4]
    svm_model = SVC(
        C=svmp["C"],
        kernel=svmp["kernel"],
        gamma=svmp["gamma"],
        probability=True,
        max_iter=10000,
        random_state=42,
    )
    svm_model.fit(all_trainset, all_trainl)
    save_path = f"{args.version}/output/{prefix_name}_{4}/{batchsize}_{lr}"
    print(f"保存地址：{save_path}")
    with open(f"{save_path}/svm_model.pkl", "wb") as file:
        pickle.dump(svm_model, file)

    log_model = LogisticRegression(C=logp["C"], random_state=42)
    log_model.fit(all_trainset, all_trainl)
    with open(f"{save_path}/log_model.pkl", "wb") as file:
        pickle.dump(log_model, file)

    knn_model = KNeighborsClassifier(
        n_neighbors=knnp["n_neighbors"],
        weights=knnp["weights"],
        p=knnp["p"],
        algorithm=knnp["algorithm"],
    )
    knn_model.fit(all_trainset, all_trainl)
    with open(f"{save_path}/knn_model.pkl", "wb") as file:
        pickle.dump(knn_model, file)

    dt_model = tree.DecisionTreeClassifier(
        criterion=dtp["criterion"],
        random_state=42,
        splitter=dtp["splitter"],
        max_depth=dtp["max_depth"],
        min_samples_split=dtp["min_samples_split"],
        class_weight="balanced",
    )
    dt_model.fit(all_trainset, all_trainl)
    with open(f"{save_path}/dt_model.pkl", "wb") as file:
        pickle.dump(dt_model, file)
    
    rf_model = RandomForestClassifier(
        n_estimators=rfp["n_estimators"],
        max_depth=rfp["max_depth"],
        max_features=rfp["max_features"],
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(all_trainset, all_trainl)
    with open(f"{save_path}/rf_model.pkl", "wb") as file:
        pickle.dump(rf_model, file)
    

def myvote(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
    print("my vote")
    svmstaestimatesub = []
    lstaestimatesub = []
    knnstaestimatesub = []
    dtstaestimatesub = []
    rfstaestimatesub = []
    videoesti = []
    participants = []
    reals = []
    mediasample = []
    allpre, allpro, records = [], [], []
    with open(f"{sample}/participant14.txt", "r") as f:
        myfile = f.read()
        myrecords = myfile.split("\n")
    for i in range(5):
        real = []
        participant = myrecords[i].split("\t")
        participants.append(participant)
        for j in range(len(participant) - 1):
            if participant[j] == "":
                pass
            elif participant[j][0] == "n":  # 新数据集是t,旧数据集是n
                real.append(0)
            else:
                real.append(1)
        reals.append(real)
        mep = []
        with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
            mef = f.read()
            mer = mef.split("\n")
            for x in range(len(mer) - 1):
                mep.append(mer[x].split("\t"))
        mediasample.append(mep)
        files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
        for file in files:
            if file[:ws] == "test_epoch" + str(bestepoch):
                name = file
                break
        with open(
            f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
        ) as fo:
            myFile = fo.read()
            myRecords = myFile.split("\n")
            subject, pre, pro1, pro0, videolabel = [], [], [], [], []
            for y in range(0, len(myRecords) - 1):
                subject.append(myRecords[y].split("\t"))  # 把表格数据存入
            for z in range(0, len(subject)):
                pre.append(int(subject[z][1]))
                pro1.append(float(subject[z][2]))
                pro0.append(1 - float(subject[z][2]))
                videolabel.append(int(subject[z][0]))
            allpre.append(pre)
            allpro.append(pro1)

    for i in range(len(reals)):
        record = []
        for j in range(len(reals[i])):
            record1 = []
            for m in range(1, 15):
                for r in range(1, 4):
                    bz = 0
                    for y in range(len(mediasample[i])):
                        if mediasample[i][y][1] == participants[i][j] + str(
                            r
                        ) and mediasample[i][y][0] == "media" + str(m):
                            bz = 1
                            record1.append(allpro[i][y])
                            break
                    if bz == 0:
                        record1.append(0.5)
            record.append(record1)
        records.append(record)

    kernel = ["rbf", "poly", "sigmoid"]
    C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    Cs = [1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    Nestimater = [3, 4, 5, 6, 7, 8, 9, 10]
    weights = ["uniform", "distance"]
    P = [1, 2, 3, 4, 5, 6]
    algorthm = ["auto", "ball_tree", "kd_tree", "brute"]

    criterion = ["entropy", "gini"]
    splitter = ["best", "random"]
    split = [2, 3, 4, 5, 6]
    maxdep = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    nemi = [5, 10, 25, 50, 100, 200]
    maxd = [10, 30, 50, 70, 90]
    maxf = [None, "sqrt", "auto", "log2"]

    trainset, trainl, valset, vall = [], [], [], []
    for i in range(5):
        valset.append(records[i])
        vall.append(reals[i])
        tras, tral = [], []
        for j in range(5):
            if j != i:
                tras.extend(records[j])
                tral.extend(reals[j])
        trainX2, trainlabel2 = tras, tral
        trainset.append(trainX2)
        trainl.append(trainlabel2)
    svmacc, logacc, knnacc, dtacc, rfacc = 0, 0, 0, 0, 0
    svmpre, svmpro, logpre, logpro, knnpre, knnpro, dtpre, dtpro, rfpre, rfpro = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    svmp, logp, knnp, dtp, rfp = {}, {}, {}, {}, {}
    for k in kernel:
        for g in gamma:
            for c in C:
                stapre1, pro1 = [], []
                for r in range(5):
                    svm = SVC(
                        C=c,
                        kernel=k,
                        gamma=g,
                        probability=True,
                        max_iter=10000,
                        random_state=42,
                    )
                    svm.fit(trainset[r], trainl[r])
                    rpre = svm.predict(valset[r])
                    rpro = svm.predict_proba(valset[r])
                    # stapre1.append(rpre)
                    pro1.append(rpro)
                    # score+=metrics.accuracy_score(rpre,vall[r])
                for yuzh in range(1, 200):
                    score = 0
                    pre = []
                    for r in range(5):
                        softlabel = []
                        for j in range(len(pro1[r])):
                            if pro1[r][j][1] >= 0.01 * yuzh:
                                softlabel.append(1)
                            else:
                                softlabel.append(0)
                        pre.append(softlabel)
                        score += metrics.accuracy_score(softlabel, vall[r])

                    score /= 5
                    if score > svmacc:
                        svmacc = score
                        svmpre = pre
                        svmpro = pro1
                        svmp = {"gamma": g, "C": c, "kernel": k, "yuzhi": yuzh}
    for k in Nestimater:
        for g in weights:
            for c in P:
                for a in algorthm:
                    score = 0
                    stapre1, pro1 = [], []
                    for r in range(5):
                        svm = KNeighborsClassifier(
                            n_neighbors=k, weights=g, p=c, algorithm=a
                        )
                        svm.fit(trainset[r], trainl[r])
                        rpre = svm.predict(valset[r])
                        rpro = svm.predict_proba(valset[r])
                        stapre1.append(rpre)
                        pro1.append(rpro)
                    for yuzh in range(1, 200):
                        score = 0
                        pre = []
                        for r in range(5):
                            softlabel = []
                            for j in range(len(pro1[r])):
                                if pro1[r][j][1] >= 0.01 * yuzh:
                                    softlabel.append(1)
                                else:
                                    softlabel.append(0)
                            pre.append(softlabel)
                            score += metrics.accuracy_score(softlabel, vall[r])
                        score /= 5
                        if score > knnacc:
                            knnacc = score
                            knnpre = pre
                            knnpro = pro1
                            knnp = {
                                "n_neighbors": k,
                                "weights": g,
                                "p": c,
                                "algorithm": a,
                                "yuzhi": yuzh,
                            }
    for k in criterion:
        for g in splitter:
            for c in split:
                for a in maxdep:
                    score = 0
                    stapre1, pro1 = [], []
                    for r in range(5):
                        svm = tree.DecisionTreeClassifier(
                            criterion=k,
                            random_state=42,
                            splitter=g,
                            max_depth=a,
                            min_samples_split=c,
                            class_weight="balanced",
                        )
                        svm.fit(trainset[r], trainl[r])
                        rpre = svm.predict(valset[r])
                        rpro = svm.predict_proba(valset[r])
                        stapre1.append(rpre)
                        pro1.append(rpro)
                    for yuzh in range(1, 200):
                        score = 0
                        pre = []
                        for r in range(5):
                            softlabel = []
                            for j in range(len(pro1[r])):
                                if pro1[r][j][1] >= 0.01 * yuzh:
                                    softlabel.append(1)
                                else:
                                    softlabel.append(0)
                            pre.append(softlabel)
                            score += metrics.accuracy_score(softlabel, vall[r])
                        score /= 5
                        if score > dtacc:
                            dtacc = score
                            dtpre = pre
                            dtpro = pro1
                            dtp = {
                                "criterion": k,
                                "splitter": g,
                                "max_depth": a,
                                "min_samples_split": c,
                                "yuzhi": yuzh,
                            }
    for k in nemi:
        for g in maxd:
            for c in maxf:
                score = 0
                stapre1, pro1 = [], []
                for r in range(5):
                    svm = RandomForestClassifier(
                        n_estimators=k,
                        max_depth=g,
                        max_features=c,
                        random_state=42,
                        class_weight="balanced",
                    )
                    svm.fit(trainset[r], trainl[r])
                    rpre = svm.predict(valset[r])
                    rpro = svm.predict_proba(valset[r])
                    stapre1.append(rpre)
                    pro1.append(rpro)
                for yuzh in range(1, 200):
                    score = 0
                    pre = []
                    for r in range(5):
                        softlabel = []
                        for j in range(len(pro1[r])):
                            if pro1[r][j][1] >= 0.01 * yuzh:
                                softlabel.append(1)
                            else:
                                softlabel.append(0)
                        pre.append(softlabel)
                        score += metrics.accuracy_score(softlabel, vall[r])
                    score /= 5
                    if score > rfacc:
                        rfacc = score
                        rfpre = pre
                        rfpro = pro1
                        rfp = {
                            "n_estimators": k,
                            "max_depth": g,
                            "max_features": c,
                            "yuzhi": yuzh,
                        }
    # print(svmpro)
    for c in Cs:
        stapre1, pro1 = [], []
        for r in range(5):
            svm = LogisticRegression(C=c, random_state=42)
            svm.fit(trainset[r], trainl[r])
            rpre = svm.predict(valset[r])
            rpro = svm.predict_proba(valset[r])
            pro1.append(rpro)
        for yuzhi in range(1, 101):
            score = 0
            pre = []
            for r in range(5):
                softlabel = []
                for j in range(len(pro1[r])):
                    if pro1[r][j][1] >= 0.01 * yuzhi:
                        softlabel.append(1)
                    else:
                        softlabel.append(0)
                pre.append(softlabel)
                score += metrics.accuracy_score(softlabel, vall[r])

            score /= 5
            if score > logacc:
                logacc = score
                logpre = pre
                logpro = pro1
                logp = {"C": c, "yuzhi": yuzhi}
    # print(logpro)
    for i in range(5):
        svmstaestimatesub.append(calestimate(vall[i], svmpre[i]))
        lstaestimatesub.append(calestimate(vall[i], logpre[i]))
        knnstaestimatesub.append(calestimate(vall[i], knnpre[i]))
        dtstaestimatesub.append(calestimate(vall[i], dtpre[i]))
        rfstaestimatesub.append(calestimate(vall[i], rfpre[i]))
    print("svm")
    print(svmp)
    meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
        args, prefix_name, svmstaestimatesub
    )
    print("logisticRegression")
    print(logp)
    _ = printresult(args, prefix_name, lstaestimatesub)
    print("knn")
    print(knnp)
    _ = printresult(args, prefix_name, knnstaestimatesub)
    print("DTree")
    print(dtp)
    _ = printresult(args, prefix_name, dtstaestimatesub)
    print("RF")
    print(rfp)
    _ = printresult(args, prefix_name, rfstaestimatesub)


def myvote_s3(args, prefix_name, batchsize, lr, sample, ws, bestepoch):
    print("my vote on sequence 3")
    # if os.path.exists(f'{args.version}/{prefix_name}/SVM'):
    #     os.makedirs(f'{args.version}/{prefix_name}/SVM')
    # if os.path.exists(f'{args.version}/{prefix_name}/LR'):
    #     os.makedirs(f'{args.version}/{prefix_name}/LR')
    # if os.path.exists(f'{args.version}/{prefix_name}/DT'):
    #     os.makedirs(f'{args.version}/{prefix_name}/DT')
    # if os.path.exists(f'{args.version}/{prefix_name}/KNN'):
    #     os.makedirs(f'{args.version}/{prefix_name}/KNN')
    # if os.path.exists(f'{args.version}/{prefix_name}/RF'):
    #     os.makedirs(f'{args.version}/{prefix_name}/RF')
    svmstaestimatesub = []
    lstaestimatesub = []
    knnstaestimatesub = []
    dtstaestimatesub = []
    rfstaestimatesub = []
    videoesti = []
    participants = []
    reals = []
    mediasample = []
    allpre, allpro, records = [], [], []
    with open(f"{sample}/participant14.txt", "r") as f:
        myfile = f.read()
        myrecords = myfile.split("\n")
    for i in range(5):
        participant = myrecords[i].split("\t")
        participants.append(participant)
        # real = []
        # for j in range(len(participant) - 1):
        #     if participant[j] == '':
        #         pass
        #     elif participant[j][0] == 'n':
        #         real.append(0)
        #     else:
        #         real.append(1)
        # reals.append(real)
        mep = []
        with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
            mef = f.read()
            mer = mef.split("\n")
            for x in range(len(mer) - 1):
                mep.append(mer[x].split("\t"))
        mediasample.append(mep)
        files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
        for file in files:
            if file[:ws] == "test_epoch" + str(bestepoch):
                name = file
                break
        with open(
            f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
        ) as fo:
            myFile = fo.read()
            myRecords = myFile.split("\n")
            subject, pre, pro1, pro0, videolabel = [], [], [], [], []
            for y in range(0, len(myRecords) - 1):
                subject.append(myRecords[y].split("\t"))  # 把表格数据存入
            for z in range(0, len(subject)):
                pre.append(int(subject[z][1]))
                pro1.append(float(subject[z][2]))
                pro0.append(1 - float(subject[z][2]))
                videolabel.append(int(subject[z][0]))
            allpre.append(pre)
            allpro.append(pro1)

    for i in range(len(participants)):
        record = []
        real = []
        for j in range(len(participants[i]) - 1):
            record1 = []
            js = 0
            for m in range(1, 15):
                bz = 0
                for y in range(len(mediasample[i])):
                    if mediasample[i][y][1] == participants[i][j] + str(
                        3
                    ) and mediasample[i][y][0] == "media" + str(m):
                        js = js + 1
                        bz = 1
                        record1.append(allpro[i][y])
                        break
                if bz == 0:
                    record1.append(0.5)
            if js > 6:
                if participants[i][j][0] == "n":
                    real.append(0)
                else:
                    real.append(1)
                record.append(record1)
        records.append(record)
        reals.append(real)

    kernel = ["rbf", "poly", "sigmoid"]
    C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    Cs = [1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    Nestimater = [3, 4, 5, 6, 7, 8, 9, 10]
    weights = ["uniform", "distance"]
    P = [1, 2, 3, 4, 5, 6]
    algorthm = ["auto", "ball_tree", "kd_tree", "brute"]

    criterion = ["entropy", "gini"]
    splitter = ["best", "random"]
    split = [2, 3, 4, 5, 6]
    maxdep = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    nemi = [5, 10, 25, 50, 100, 200]
    maxd = [10, 30, 50, 70, 90]
    maxf = [None, "sqrt", "auto", "log2"]

    trainset, trainl, valset, vall = [], [], [], []
    for i in range(5):
        valset.append(records[i])
        vall.append(reals[i])
        tras, tral = [], []
        for j in range(5):
            if j != i:
                tras.extend(records[j])
                tral.extend(reals[j])
        trainX2, trainlabel2 = tras, tral
        trainset.append(trainX2)
        trainl.append(trainlabel2)
    svmacc, logacc, knnacc, dtacc, rfacc = 0, 0, 0, 0, 0
    svmpre, svmpro, logpre, logpro, knnpre, knnpro, dtpre, dtpro, rfpre, rfpro = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    svmp, logp, knnp, dtp, rfp = {}, {}, {}, {}, {}
    for k in kernel:
        for g in gamma:
            for c in C:
                stapre1, pro1 = [], []
                for r in range(5):
                    svm = SVC(
                        C=c,
                        kernel=k,
                        gamma=g,
                        probability=True,
                        max_iter=10000,
                        random_state=42,
                    )
                    svm.fit(trainset[r], trainl[r])
                    rpre = svm.predict(valset[r])
                    rpro = svm.predict_proba(valset[r])
                    # stapre1.append(rpre)
                    pro1.append(rpro)
                    # score+=metrics.accuracy_score(rpre,vall[r])
                for yuzh in range(1, 101):
                    score = 0
                    pre = []
                    for r in range(5):
                        softlabel = []
                        for j in range(len(pro1[r])):
                            if pro1[r][j][1] >= 0.01 * yuzh:
                                softlabel.append(1)
                            else:
                                softlabel.append(0)
                        pre.append(softlabel)
                        score += metrics.accuracy_score(softlabel, vall[r])

                    score /= 5
                    if score > svmacc:
                        svmacc = score
                        svmpre = pre
                        svmpro = pro1
                        svmp = {"gamma": g, "C": c, "kernel": k, "yuzhi": yuzh}

    for k in Nestimater:
        for g in weights:
            for c in P:
                for a in algorthm:
                    score = 0
                    stapre1, pro1 = [], []
                    for r in range(5):
                        svm = KNeighborsClassifier(
                            n_neighbors=k, weights=g, p=c, algorithm=a
                        )
                        svm.fit(trainset[r], trainl[r])
                        rpre = svm.predict(valset[r])
                        rpro = svm.predict_proba(valset[r])
                        stapre1.append(rpre)
                        pro1.append(rpro)
                    for yuzh in range(1, 200):
                        score = 0
                        pre = []
                        for r in range(5):
                            softlabel = []
                            for j in range(len(pro1[r])):
                                if pro1[r][j][1] >= 0.01 * yuzh:
                                    softlabel.append(1)
                                else:
                                    softlabel.append(0)
                            pre.append(softlabel)
                            score += metrics.accuracy_score(softlabel, vall[r])
                        score /= 5
                        if score > knnacc:
                            knnacc = score
                            knnpre = pre
                            knnpro = pro1
                            knnp = {
                                "n_neighbors": k,
                                "weights": g,
                                "p": c,
                                "algorithm": a,
                                "yuzhi": yuzh,
                            }
    for k in criterion:
        for g in splitter:
            for c in split:
                for a in maxdep:
                    score = 0
                    stapre1, pro1 = [], []
                    for r in range(5):
                        svm = tree.DecisionTreeClassifier(
                            criterion=k,
                            random_state=42,
                            splitter=g,
                            max_depth=a,
                            min_samples_split=c,
                            class_weight="balanced",
                        )
                        svm.fit(trainset[r], trainl[r])
                        rpre = svm.predict(valset[r])
                        rpro = svm.predict_proba(valset[r])
                        stapre1.append(rpre)
                        pro1.append(rpro)
                    for yuzh in range(1, 200):
                        score = 0
                        pre = []
                        for r in range(5):
                            softlabel = []
                            for j in range(len(pro1[r])):
                                if pro1[r][j][1] >= 0.01 * yuzh:
                                    softlabel.append(1)
                                else:
                                    softlabel.append(0)
                            pre.append(softlabel)
                            score += metrics.accuracy_score(softlabel, vall[r])
                        score /= 5
                        if score > dtacc:
                            dtacc = score
                            dtpre = pre
                            dtpro = pro1
                            dtp = {
                                "criterion": k,
                                "splitter": g,
                                "max_depth": a,
                                "min_samples_split": c,
                                "yuzhi": yuzh,
                            }
    for k in nemi:
        for g in maxd:
            for c in maxf:
                score = 0
                stapre1, pro1 = [], []
                for r in range(5):
                    svm = RandomForestClassifier(
                        n_estimators=k,
                        max_depth=g,
                        max_features=c,
                        random_state=42,
                        class_weight="balanced",
                    )
                    svm.fit(trainset[r], trainl[r])
                    rpre = svm.predict(valset[r])
                    rpro = svm.predict_proba(valset[r])
                    stapre1.append(rpre)
                    pro1.append(rpro)
                for yuzh in range(1, 200):
                    score = 0
                    pre = []
                    for r in range(5):
                        softlabel = []
                        for j in range(len(pro1[r])):
                            if pro1[r][j][1] >= 0.01 * yuzh:
                                softlabel.append(1)
                            else:
                                softlabel.append(0)
                        pre.append(softlabel)
                        score += metrics.accuracy_score(softlabel, vall[r])
                    score /= 5
                    if score > rfacc:
                        rfacc = score
                        rfpre = pre
                        rfpro = pro1
                        rfp = {
                            "n_estimators": k,
                            "max_depth": g,
                            "max_features": c,
                            "yuzhi": yuzh,
                        }
    # print(svmpro)
    for c in Cs:
        stapre1, pro1 = [], []
        for r in range(5):
            svm = LogisticRegression(C=c, random_state=42)
            svm.fit(trainset[r], trainl[r])
            rpre = svm.predict(valset[r])
            rpro = svm.predict_proba(valset[r])
            pro1.append(rpro)
        for yuzhi in range(1, 101):
            score = 0
            pre = []
            for r in range(5):
                softlabel = []
                for j in range(len(pro1[r])):
                    if pro1[r][j][1] >= 0.01 * yuzhi:
                        softlabel.append(1)
                    else:
                        softlabel.append(0)
                pre.append(softlabel)
                score += metrics.accuracy_score(softlabel, vall[r])

            score /= 5
            if score > logacc:
                logacc = score
                logpre = pre
                logpro = pro1
                logp = {"C": c, "yuzhi": yuzhi}
    # print(logpro)
    for i in range(5):
        svmstaestimatesub.append(calestimate(vall[i], svmpre[i]))
        lstaestimatesub.append(calestimate(vall[i], logpre[i]))
        knnstaestimatesub.append(calestimate(vall[i], knnpre[i]))
        dtstaestimatesub.append(calestimate(vall[i], dtpre[i]))
        rfstaestimatesub.append(calestimate(vall[i], rfpre[i]))
    print("svm")
    print(svmp)
    meanaccdv, meanprecisiondv, meanrecalldv, meanf1dv, meanspecificty = printresult(
        args, prefix_name, svmstaestimatesub
    )
    print("logisticRegression")
    print(logp)
    _ = printresult(args, prefix_name, lstaestimatesub)
    print("knn")
    print(knnp)
    _ = printresult(args, prefix_name, knnstaestimatesub)
    print("DTree")
    print(dtp)
    _ = printresult(args, prefix_name, dtstaestimatesub)
    print("RF")
    print(rfp)
    _ = printresult(args, prefix_name, rfstaestimatesub)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def printpro(args, prefix_name, sample, i, batchsize, lr):
    mep = []
    fmep = []
    allpre = []
    allpro = []
    real = []
    with open(f"{sample}/participant14.txt", "r") as f:
        myfile = f.read()
        records = myfile.split("\n")
        participants = records[i].split("\t")
        for j in range(len(participants) - 1):
            if participants[j] == "":
                print("!")
                pass
            elif participants[j][0] == "n":
                real.append(0)
            else:
                real.append(1)
    with open(f"{sample}/14/{i}/testsample.txt", "r") as f:
        mef = f.read()
        mer = mef.split("\n")
        for x in range(len(mer) - 1):
            mep.append(mer[x].split("\t"))
    #########################
    # for k in range(len(mep)):
    #     list = os.listdir(f'{imgpath}/{mep[k][0]}')
    #     # samplesaug.append(samples[i])
    #     for j in range(len(list)):
    #         if mep[k][1] == list[j].split('.')[0]:
    #             # for l in leng:
    #             #     if l[0][:len(participants[i][k])] == participants[i][k] and len(l[2]) != 1:
    #             fmep.append(mep[k])
    #             break
    # mediasample.extend(fmep)
    #########################
    mediasample = mep
    print(len(participants), len(real))
    files = os.listdir(f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}")
    for file in files:
        if file[:12] == "test_epoch20":
            name = file
            # break
    with open(
        f"{args.version}/output/{prefix_name}_{i}/{batchsize}_{lr}/{name}", "r"
    ) as fo:
        myFile = fo.read()
        myRecords = myFile.split("\n")
        subject, pre, pro = [], [], []
        for y in range(0, len(myRecords) - 1):
            subject.append(myRecords[y].split("\t"))  # 把表格数据存入
        for z in range(0, len(subject)):
            pre.append(int(subject[z][1]))
            pro.append(float(subject[z][2]))
        allpre.extend(pre)
        allpro.extend(pro)
    shuchu = [
        [
            "participant",
            "label",
            "media1",
            "media2",
            "media3",
            "media4",
            "media5",
            "media6",
            "media7",
            "media8",
            "media9",
            "media10",
            "media11",
            "media12",
            "media13",
            "media14",
        ]
    ]
    for m in range(len(participants) - 1):
        for n in range(1, 4):
            rec = [participants[m], real[m]]
            for j in range(1, 15):
                pr = np.nan
                for k in range(len(mediasample)):
                    if mediasample[k][0] == "media" + str(j) and mediasample[k][
                        1
                    ] == participants[m] + str(n):
                        pr = allpro[k]
                rec.append(pr)
            shuchu.append(rec)
    f = xlwt.Workbook()
    sheet3 = f.add_sheet("sheet1", cell_overwrite_ok=True)  # 创建sheet
    for k in range(len(shuchu)):
        for j in range(len(shuchu[k])):
            sheet3.write(k, j, shuchu[k][j])
    prefix_name = prefix_name.replace("/", "_")
    f.save(f"{args.version}_{prefix_name}_{i}_{batchsize}_{lr}.xls")


def statistic(args, prefix_name, rou, batchsize, lr, sample, ws, bestepoch):
    fo = open(f"{sample}/14/{rou}/testsample.txt", "r")
    participants = []
    myFile = fo.read()
    myRecords = myFile.split("\n")
    for y in range(0, len(myRecords) - 1):
        participants.append(myRecords[y].split("\t"))  # 把表格数据存入
    # for i in range(len(participants)):
    #     pre(imgpath,resultpath,participants[i],net)
    #
    # # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_0/32_1e-06/test_epoch100_trainacc0.8901_valacc0.6155.txt', 'r')
    # # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_1/32_1e-06/test_epoch100_trainacc0.9006_valacc0.6019.txt', 'r')
    # # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_2/32_1e-06/test_epoch100_trainacc0.8842_valacc0.6393.txt', 'r')
    # # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_3/32_1e-06/test_epoch100_trainacc0.8942_valacc0.6321.txt', 'r')
    files = os.listdir(f"{args.version}/output/{prefix_name}_{rou}/{batchsize}_{lr}")
    for file in files:
        if file[:ws] == "test_epoch" + str(bestepoch):
            name = file
            break
    fo = open(f"{args.version}/output/{prefix_name}_{rou}/{batchsize}_{lr}/{name}", "r")
    val = []
    myFile = fo.read()
    myRecords = myFile.split("\n")
    for y in range(0, len(myRecords) - 1):
        val.append(myRecords[y].split("\t"))  # 把表格数据存入

    sequence = pd.read_excel("../DataProcessing/视频播放顺序.xlsx")
    sequence = sequence.values.tolist()
    sequence = np.delete(sequence, [37, 38], 0)
    s1, s2, s3 = [], [], []
    for k in range(len(val)):
        for i in range(len(sequence)):
            if participants[k][1][:-1] == sequence[i][0]:
                if sequence[i][int(participants[k][1][-1])] == str(1.0):
                    s1.append(participants[k])
                    break
                elif sequence[i][int(participants[k][1][-1])] == str(2.0):
                    s2.append(participants[k])
                    break
                elif sequence[i][int(participants[k][1][-1])] == str(3.0):
                    s3.append(participants[k])
                    break

    samples = len(participants)
    sampleright = 0
    samplewrong = 0
    allASD = 0
    allTD = 0
    ASDright = 0
    TDright = 0
    ASDwrong = 0
    TDwrong = 0
    rou1right = 0
    rou2right = 0
    rou3right = 0

    mediasample = {}
    mediaASD = {}
    mediaTD = {}
    mediawrong = {}
    mediaright = {}
    mediaASDwrong = {}
    mediaASDright = {}
    mediaTDwrong = {}
    mediaTDright = {}
    roundsamples = {}
    roundTD = {}
    roundASD = {}
    roundTDright = {}
    roundTDwrong = {}
    roundASDright = {}
    roundASDwrong = {}
    for i in range(1, 4):
        roundsamples[str(i)] = 0
        roundTD[str(i)] = 0
        roundASD[str(i)] = 0
        roundASDwrong[str(i)] = 0
        roundASDright[str(i)] = 0
        roundTDwrong[str(i)] = 0
        roundTDright[str(i)] = 0
    for i in range(1, 15):
        mediasample["media" + str(i)] = 0
        mediaASDright["media" + str(i)] = 0
        mediawrong["media" + str(i)] = 0
        mediaTDwrong["media" + str(i)] = 0
        mediaright["media" + str(i)] = 0
        mediaASDwrong["media" + str(i)] = 0
        mediaTD["media" + str(i)] = 0
        mediaASD["media" + str(i)] = 0
        mediaTDright["media" + str(i)] = 0
    # for i in range(len(participants)):
    #     mediasample[participants[i][0]]+=1
    #     roundsamples[participants[i][1][-1]]+=1
    #     ASD,right=pre(imgpath,resultpath,participants[i],net)
    #     if right=='right':
    #         sampleright+=1
    #     else:
    #         samplewrong+=1
    #     if ASD=='ASD':
    #         allASD+=1
    #         mediaASD[participants[i][0]]+=1
    #         roundASD[participants[i][1][-1]]+=1
    #         if right=='right':
    #             ASDright+=1
    #             mediaASDright[participants[i][0]]+=1
    #             roundASDright[participants[i][1][-1]]+=1
    #         else:
    #             ASDwrong+=1
    #             mediaASDwrong[participants[i][0]]+=1
    #             roundASDwrong[participants[i][1][-1]]+=1
    #     else:
    #         allTD+=1
    #         mediaTD[participants[i][0]]+=1
    #         roundTD[participants[i][1][-1]]+=1
    #         if right=='right':
    #             TDright+=1
    #             mediaTDright[participants[i][0]]+=1
    #             roundTDright[participants[i][1][-1]]+=1
    #         else:
    #             TDwrong+=1
    #             mediaTDwrong[participants[i][0]]+=1
    #             roundTDwrong[participants[i][1][-1]]+=1

    for i in range(len(participants)):
        mediasample[participants[i][0]] += 1
        roundsamples[participants[i][1][-1]] += 1

        if val[i][0] == str(0):
            ASD = "TD"
        else:
            ASD = "ASD"
        if val[i][0] == val[i][1]:
            right = "right"
        else:
            right = "wrong"

        if right == "right":
            sampleright += 1
            mediaright[participants[i][0]] += 1
            bz = 0
            for j in range(len(s1)):
                if s1[j][1] == participants[i][1]:
                    rou1right += 1
                    bz = 1
                    break
            if bz == 0:
                for j in range(len(s2)):
                    if s2[j][1] == participants[i][1]:
                        rou2right += 1
                        break
            if bz == 0:
                for j in range(len(s3)):
                    if s3[j][1] == participants[i][1]:
                        rou3right += 1
                        break

        else:
            samplewrong += 1
        if ASD == "ASD":
            allASD += 1
            mediaASD[participants[i][0]] += 1
            roundASD[participants[i][1][-1]] += 1
            if right == "right":
                ASDright += 1
                mediaASDright[participants[i][0]] += 1
                roundASDright[participants[i][1][-1]] += 1
            else:
                ASDwrong += 1
                mediaASDwrong[participants[i][0]] += 1
                roundASDwrong[participants[i][1][-1]] += 1
        else:
            allTD += 1
            mediaTD[participants[i][0]] += 1
            roundTD[participants[i][1][-1]] += 1
            if right == "right":
                TDright += 1
                mediaTDright[participants[i][0]] += 1
                roundTDright[participants[i][1][-1]] += 1
            else:
                TDwrong += 1
                mediaTDwrong[participants[i][0]] += 1
                roundTDwrong[participants[i][1][-1]] += 1

    # print(f'samples: {samples}| ASD: {allASD}| TD: {allTD}')
    # print(f'round 1: {roundsamples[str(1)]}| round 2: {roundsamples[str(2)]}| round 3: {roundsamples[str(3)]}')
    # print(f'right sample: {sampleright} (right/all {np.round(sampleright/samples,4)})')
    # # print(f'wrong sample: {samplewrong} (wrong/all {np.round(samplewrong/samples,4)})')
    # print(f'right ASD sample: {ASDright} (ASD:right/all {np.round(ASDright/allASD,4)})')
    # print(f'right TD sample: {TDright} (TD:right/all {np.round(TDright/allTD,4)})')
    # # print(f'wrong ASD sample: {ASDwrong} (ASD:wrong/all {np.round(ASDwrong/allASD,4)})')
    # # print(f'wrong TD sample: {TDwrong} (TD:wrong/all {np.round(TDwrong/allTD,4)})')
    # for i in range(1,4):
    #     print(f'round{i}: ASD {roundASD[str(i)]}({np.round(roundASD[str(i)]/roundsamples[str(i)],4)})| ASDringht {roundASDright[str(i)]}({np.round(roundASDright[str(i)]/roundASD[str(i)],4)})|TD {roundTD[str(i)]}({np.round(roundTD[str(i)]/roundsamples[str(i)],4)})| TDringht {roundTDright[str(i)]}({np.round(roundTDright[str(i)]/roundTD[str(i)],4)})')
    print(rou)
    for i in range(1, 15):
        media = "media" + str(i)
        print(
            f"{media}: samples {mediasample[media]}|Acc:{np.round(mediaright[media] / mediasample[media], 4)} ASD {mediaASD[media]}| ASDringht {mediaASDright[media]}({np.round(mediaASDright[media] / mediaASD[media], 4)})|TD {mediaTD[media]}| TDringht {mediaTDright[media]}({np.round(mediaTDright[media] / mediaTD[media], 4)})"
        )
    print(f"sequence1: samples {len(s1)}| Acc:{np.round(rou1right / len(s1), 4)}")
    print(f"sequence2: samples {len(s2)}| Acc:{np.round(rou2right / len(s2), 4)}")
    print(f"sequence3: samples {len(s3)}| Acc:{np.round(rou3right / len(s3), 4)}")


setup_seed(3407)

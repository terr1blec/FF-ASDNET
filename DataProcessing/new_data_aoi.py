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
import random
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import os
from matplotlib import pyplot as plt
import numpy as np
import xlwt
import torch
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sklearn
from sklearn import tree
import pandas as pd
import pickle
import warnings

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn


from sklearn.neighbors import (
    KNeighborsClassifier,
    NeighborhoodComponentsAnalysis,
    NearestCentroid,
)

warnings.filterwarnings("ignore")


def load_data(path):
    df = pd.DataFrame()  # 初始化一个空的DataFrame
    for media in range(1, 15):
        # 读取每个文件
        file_path = os.path.join(path, "media" + str(media) + ".xls")
        df_temp = pd.read_excel(file_path)
        
        # 添加一个新列 'media_id' 表示数据来源的media文件编号
        df_temp['media'] = media
        
        # 合并到主DataFrame中
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
        
    return df

def test_new_data(model,data):
    X = data.iloc[:, 1:-2]
    y = data['label']
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y = y.to_numpy() if isinstance(y, pd.Series) else y
    
    participants_list = data.iloc[:, 0].to_numpy()
    medias_list = data.iloc[:, -2].to_numpy()

    testpa_list = [participant[:-1] for participant in participants_list]
    order_list = [participant[-1] for participant in participants_list]

    pre_list = model.predict(X)
    probas_list = model.predict_proba(X)[:, 1]
    trues_list = y # 真实标签列表

    accuracy = metrics.accuracy_score(pre_list, trues_list)
    precision = metrics.precision_score(pre_list, trues_list)
    recall = metrics.recall_score(pre_list, trues_list)
    f1 = metrics.f1_score(pre_list, trues_list)

    print("accuaty:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("auc:", metrics.roc_auc_score(trues_list, probas_list))

    return (
        accuracy,
        testpa_list,
        trues_list,
        pre_list,
        probas_list,
        medias_list,
        order_list,
    )

def plot_auc(fpr_tpr):
    #创建画布
    fig,ax = plt.subplots(figsize=(10,8))
    #自定义标签名称label=''
    ax.plot(fpr_tpr[0][0],fpr_tpr[0][1],linewidth=2,
            label='Hard vote (AUC={})'.format(str(fpr_tpr[0][2])))
    ax.plot(fpr_tpr[1][0],fpr_tpr[1][1],linewidth=2,
            label='Soft vote (AUC={})'.format(str(fpr_tpr[1][2])))
    ax.plot(fpr_tpr[2][0],fpr_tpr[2][1],linewidth=2,
            label='Soft Voting with Threshold (AUC={})'.format(str(fpr_tpr[2][2])))
    ax.plot(fpr_tpr[4][0],fpr_tpr[4][1],linewidth=2,
            label='Logistic (AUC={})'.format(str(fpr_tpr[4][2])))
    ax.plot(fpr_tpr[5][0],fpr_tpr[5][1],linewidth=2,
            label='Random forest (AUC={})'.format(str(fpr_tpr[5][2])))
    ax.plot(fpr_tpr[3][0],fpr_tpr[3][1],linewidth=2,
            label='SVM (AUC={})'.format(str(fpr_tpr[3][2])))
    #绘制对角线
    ax.plot([0,1],[0,1],linestyle='--',color='grey')
    #调整字体大小
    plt.legend(fontsize=12)
    plt.savefig('roc_auc.png', dpi=300)

def load_model(i,load_model_path):
    with open(f"{load_model_path}/{i}/svm_model_segment.pkl", "rb") as file:
        model = pickle.load(file)
    print("Successfully load!")
    return model


# 输入真实标签和预测标签，求性能指标
def calculate_performance(true_labels, predicted_labels,proba_list):
    """
    计算性能指标
    """
    auc = metrics.roc_auc_score(true_labels, proba_list)
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels)
    recall = metrics.recall_score(true_labels, predicted_labels)
    f1 = metrics.f1_score(true_labels, predicted_labels)
    tn, fp, fn, tp = metrics.confusion_matrix(true_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)
    return [auc, accuracy, precision, recall, f1, specificity,auc]


# 输入已经计算好的各折性能指标，计算均值和方差，并打印
def print_result(estimate):
    """
    打印性能指标的平均值和方差
    """
    metrics = ["accuracy", "precision", "recall", "f1", "specificity","auc"]
    results = []
    for i in range(len(estimate)):
        results.append([estimate[i][j] for j in range(1, 7)])
    mean_results = np.mean(results, axis=0)
    mean_results = np.round(mean_results, 4)
    var_results = np.std(results, axis=0)
    var_results = np.round(var_results, 4)
    for i in range(len(metrics)):
        print(f"{metrics[i]}: {mean_results[i]} +/- {var_results[i]}")
    return (
        mean_results[0],
        mean_results[1],
        mean_results[2],
        mean_results[3],
        mean_results[4],
        mean_results[5],
    )


def hard_vote(testpa_list, probas_list):
    """
    硬投票
    """
    participants = set(testpa_list)
    subject_pre = []
    subject_pro = []
    for participant in participants:
        index = [i for i, x in enumerate(testpa_list) if x == participant]
        probas = [probas_list[i] for i in index]
        pre = [1 if proba > 0.5 else 0 for proba in probas]
        subject_pro.append(sum(pre) / len(pre))
        pre = 1 if sum(pre) > len(pre) / 2 else 0
        subject_pre.append(pre)


    print(f"一共有{len(testpa_list)}个测试样本")
    print(f"一共有{len(participants)}个参与者")
    subject_true = [0 if x[0] == "t" else 1 for x in participants]
    count_ones = subject_true.count(0)
    print(f"其中有{count_ones}个TD，{len(participants) - count_ones}个ASD")
    print(f"hard vote acc: {metrics.accuracy_score(subject_true, subject_pre)}")
    estimate = calculate_performance(subject_true, subject_pre,subject_pro)
    hard_fpr_tpr = []
    fpr_hard,tpr_hard,_ = metrics.roc_curve(subject_true,subject_pro)
    hard_auc = round(metrics.roc_auc_score(subject_true, subject_pro),4)
    hard_fpr_tpr.append([fpr_hard,tpr_hard,hard_auc])

    return estimate,hard_fpr_tpr


def soft_vote(testpa_list, probas_list):
    """
    软投票
    """
    participants = set(testpa_list)
    subject_pre = []
    subject_pro = []
    for participant in participants:
        index = [i for i, x in enumerate(testpa_list) if x == participant]
        probas = [probas_list[i] for i in index]
        pre = sum(probas) / len(probas)
        subject_pro.append(pre)
        pre = 1 if pre > 0.5 else 0
        subject_pre.append(pre)
        

    subject_true = [0 if x[0] == "t" else 1 for x in participants]
    print(f"soft vote acc: {metrics.accuracy_score(subject_true, subject_pre)}")
    estimate = calculate_performance(subject_true, subject_pre,subject_pro)
    soft_fpr_tpr = []
    fpr_soft,tpr_soft,_ = metrics.roc_curve(subject_true,subject_pro)
    soft_auc = round(metrics.roc_auc_score(subject_true, subject_pro),4)
    soft_fpr_tpr.append([fpr_soft,tpr_soft,soft_auc])
    return estimate,soft_fpr_tpr


def best_soft_vote(testpa_list, probas_list):
    """
    最佳软投票
    """
    participants = set(testpa_list)
    subject_true = [0 if x[0] == "t" else 1 for x in participants]
    bestacc = 0
    thresholds = 52
    subject_pre = []
    subject_pro = []
    for participant in participants:
        index = [i for i, x in enumerate(testpa_list) if x == participant]
        probas = [probas_list[i] for i in index]
        pre = sum(probas) / len(probas)
        subject_pro.append(min(pre+ (50-thresholds)*0.01,1))
        pre = 1 if pre > thresholds * 0.01 else 0
        subject_pre.append(pre)

    print(f"best thresholds: {thresholds}")
    print(
        f"best soft vote acc: {metrics.accuracy_score(subject_true, subject_pre)}"
    )
    soft_fpr_tpr = []
    estimate = calculate_performance(subject_true, subject_pre,subject_pro)
    fpr_soft,tpr_soft,_ = metrics.roc_curve(subject_true,subject_pro)
    soft_auc = round(metrics.roc_auc_score(subject_true, subject_pro),4)
    soft_fpr_tpr.append([fpr_soft,tpr_soft,soft_auc])
    return estimate,soft_fpr_tpr


# def extract_number(s):
#     return int("".join([n for n in s if n.isdigit()]))


def ml_vote(testpa_list, probas_list, medias_list, order_list, ml_vote_path):
    """
    机器学习投票
    """
    participants = set(testpa_list)
    participants = sorted(participants)
    subject_true = [0 if x[0] == "t" else 1 for x in participants]
    # medias = set(medias_list)
    # orders = set(order_list)
    record = []
    for participant in participants:
        record1 = []
        for m in range(1, 15):
            for r in range(1, 4):
                bz = 0
                for i in range(len(testpa_list)):
                    if (
                        testpa_list[i] == participant
                        and medias_list[i] == m
                        and order_list[i] == str(r)
                    ):
                        bz = 1
                        record1.append(probas_list[i])
                        break
                if bz == 0:
                    record1.append(0.5)

        record.append(record1)
    
    ml_fpr_tpr = []
    with open(f"{ml_vote_path}/svm_model.pkl", "rb") as file:
        svm = pickle.load(file)
    svm_list = svm.predict(record)
    svm_pro = svm.predict_proba(record)[:,1]
    print(svm_pro)
    fpr_svm,tpr_svm,_ = metrics.roc_curve(subject_true,svm_pro)
    print(f"svm acc: {metrics.accuracy_score(subject_true, svm_list)}")
    svm_auc = round(metrics.roc_auc_score(subject_true, svm_pro),4)
    svm_estimate = calculate_performance(subject_true, svm_list,svm_pro)
    ml_fpr_tpr.append([fpr_svm,tpr_svm,svm_auc])

    with open(f"{ml_vote_path}/log_model.pkl", "rb") as file:
        log = pickle.load(file)
    log_list = log.predict(record)
    log_pro = log.predict_proba(record)[:,1]
    print(f"log acc: {metrics.accuracy_score(subject_true, log_list)}")
    log_estimate = calculate_performance(subject_true, log_list,log_pro)
    fpr_log,tpr_log,_ = metrics.roc_curve(subject_true,log_pro)
    log_auc = round(metrics.roc_auc_score(subject_true, log_pro),4)
    ml_fpr_tpr.append([fpr_log,tpr_log,log_auc])

    with open(f"{ml_vote_path}/knn_model.pkl", "rb") as file:
        knn = pickle.load(file)
    knn_list = knn.predict(record)
    knn_pro = knn.predict_proba(record)[:,1]
    print(f"knn acc: {metrics.accuracy_score(subject_true, knn_list)}")
    knn_estimate = calculate_performance(subject_true, knn_list,knn_pro)

    with open(f"{ml_vote_path}/dt_model.pkl", "rb") as file:
        dt = pickle.load(file)
    dt_list = dt.predict(record)
    dt_pro = dt.predict_proba(record)[:,1]
    print(f"dicision tree acc: {metrics.accuracy_score(subject_true, dt_list)}")
    dt_estimate = calculate_performance(subject_true, dt_list,dt_pro)

    with open(f"{ml_vote_path}/rf_model.pkl", "rb") as file:
        rf = pickle.load(file)
    rf_list = rf.predict(record)
    rf_pro = rf.predict_proba(record)[:,1]
    print(f"random forest acc: {metrics.accuracy_score(subject_true, rf_list)}")
    rf_estimate = calculate_performance(subject_true, rf_list,rf_pro)
    fpr_rf,tpr_rf,_ = metrics.roc_curve(subject_true,rf_pro)
    rf_auc = round(metrics.roc_auc_score(subject_true, rf_pro),4)
    ml_fpr_tpr.append([fpr_rf,tpr_rf,rf_auc])

    return svm_estimate, log_estimate, knn_estimate, dt_estimate, rf_estimate,ml_fpr_tpr


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def data_processing(df):
    columns_to_keep = np.r_[0:17, 19:26,-1]  # 注意Python中的切片是前闭后开的，所以是0:18而不是0:17
    df = df.iloc[:, columns_to_keep]
    # 先为所有行创建'label'列，并初始化为ASD，即1
    df['label'] = 1
    # 然后根据'sample'列的条件更新'label'列的值
    df.loc[df['sample'].str.startswith('t'), 'label'] = 0 # 以'n'开头的样本的为TD
    # 选择要缩放的列（这里假设除了'sample'和'label'列之外的所有列都是数值型特征）
    features_to_scale = df.columns.drop(['sample', 'label','media'])
    # 初始化MinMaxScaler
    scaler = MinMaxScaler()
    # 对特征进行缩放
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df

if __name__ == "__main__":
    seed = 3407
    seed_everything(seed)

    load_model_path = "model"
    ml_vote_path = "model/method"
    # ml_vote_path = "v20240402/dv_zssa_l/output/l_5fc_padding_d0.5_convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle_4/64_5e-07"

    sample = "/mnt/shareEEx/wangzilin/new_data/ASDeyetracking/DataProcessing/stafeatures/aoiwithduration"
    # I = [0, 1, 2, 3, 4]
    I = [2]
    total_acc = 0
    video_result = []
    hardvote_result = []
    sortvote_result = []
    bestsortvote_result = []
    svm_result, log_result, knn_result, dt_result, rf_result = [], [], [], [], []
    
    records = []
    data = load_data(sample)
    data = data_processing(data)
    for i in I:
        print(f"第{i}轮测试")
        
        model = load_model(i,load_model_path)

        (
            accuracy,
            testpa_list,
            trues_list,
            pre_list,
            probas_list,
            medias_list,
            order_list,
        ) = test_new_data( model,data
        )

        total_acc += accuracy
        video_result.append(calculate_performance(trues_list, pre_list, probas_list))
        hard_estimate,hard_fpr_tpr = hard_vote(testpa_list, probas_list)
        hardvote_result.append(hard_estimate)
        soft_estimate,soft_fpr_tpr = soft_vote(testpa_list, probas_list)
        sortvote_result.append(soft_estimate)
        bestsort_estimate,bestsort_fpr_tpr = best_soft_vote(testpa_list, probas_list)
        bestsortvote_result.append(bestsort_estimate)
        svm_estimate, log_estimate, knn_estimate, dt_estimate, rf_estimate,ml_fpr_tpr = (
            ml_vote(
                testpa_list, probas_list, medias_list, order_list, ml_vote_path
            )
        )
        svm_result.append(svm_estimate)
        log_result.append(log_estimate)
        knn_result.append(knn_estimate)
        dt_result.append(dt_estimate)
        rf_result.append(rf_estimate)


    total_acc = total_acc / 5
    print(f"CV Acc: {np.round(total_acc, 4)}")
    print(f"Video result:")
    print_result(video_result)
    print(f"Hard vote result:")
    print_result(hardvote_result)
    print(f"Soft vote result:")
    print_result(sortvote_result)
    print(f"Best soft vote result:")
    print_result(bestsortvote_result)
    print(f"SVM result:")
    print_result(svm_result)
    print(f"Logistic result:")
    print_result(log_result)
    print(f"Knn result:")
    print_result(knn_result)
    print(f"Decision Tree result:")
    print_result(dt_result)
    print(f"Random Forest result:")
    print_result(rf_result)
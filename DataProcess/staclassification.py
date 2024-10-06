import os
import numpy as np
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    NeighborhoodComponentsAnalysis,
    NearestCentroid,
)
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB,
    CategoricalNB,
)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn
from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings("ignore")


def presample(samplepath, is_train, rou):
    fo = open(f"{samplepath}/14/{rou}/{is_train}sample.txt", "r")
    train = []
    myFile = fo.read()
    myRecords = myFile.split("\n")
    for y in range(0, len(myRecords) - 1):
        train.append(myRecords[y].split("\t"))  # 把表格数据存入
    return train


def premediasample(samplepath, is_train, rou, media):
    fo = open(f"{samplepath}/14/{rou}/{is_train}sample.txt", "r")
    train = []
    sample = []
    myFile = fo.read()
    myRecords = myFile.split("\n")
    for y in range(0, len(myRecords) - 1):
        train.append(myRecords[y].split("\t"))  # 把表格数据存入

    for i in range(len(train)):
        if train[i][0] == "media" + str(media):
            sample.append(train[i][1])
    return sample


def predata(sample, featurespath):
    asd = 0
    td = 0
    allfeatures = {}
    for media in range(1, 15):
        features1 = pd.read_excel(f"{featurespath}/media{media}.xls")
        features2 = features1.values.tolist()
        # features=[fea[0:17]+fea[19:21] for fea in features2]
        features = [
            [fea[0:6] + fea[8:11] + fea[13:17] + [fea[19]] + fea[21:23]]
            for fea in features2
        ]
        allfeatures["media" + str(media)] = features
    refeatures = []
    label = []
    for i in range(len(sample)):
        e = 0
        for j in range(len(allfeatures[sample[i][0]])):
            if allfeatures[sample[i][0]][j][0] == sample[i][1]:
                e = 1
                refeatures.append(allfeatures[sample[i][0]][j][1:])
                if sample[i][1][0] == "n":
                    label.append(0)
                    td += 1
                else:
                    label.append(1)
                    asd += 1
        if e == 0:
            print(f"{sample[i][0]} do not have {sample[i][1]}")

    pass
    return refeatures, label, asd, td


def premediadata(sample, featurespath, media):
    asd = 0
    td = 0
    features1 = pd.read_excel(f"{featurespath}/media{media}.xls")
    features2 = features1.values.tolist()
    # features=[fea[0:3]+fea[5:15]+fea[19:21] for fea in features2]
    # features=[fea[0:3]+fea[5:15]+fea[19:21] for fea in features2]
    # features = [fea[0:21] for fea in features2]
    # features=[fea[0:17]+fea[19:21] for fea in features2]
    # features = [fea[0:6] + fea[8:11] + fea[13:17] + [fea[19]]+fea[21:23] for fea in features2]
    features = [fea[0:17] + fea[19:26] for fea in features2]
    refeatures = []
    label = []
    resample = []
    for i in range(len(sample)):
        e = 0
        for j in range(len(features)):
            if features[j][0] == sample[i]:
                e = 1
                refeatures.append(features[j][1:])
                resample.append(features[j][0])
                if sample[i][0] == "n":
                    label.append(0)
                    td += 1
                else:
                    label.append(1)
                    asd += 1
        # if e==0:
        #     print(f'{sample[i][0]} do not have {sample[i][1]}')

    pass
    return resample, refeatures, label, asd, td


def svmtiaocan(C, gamma, kernel):
    best_score = 0
    for k in kernel:
        for g in gamma:
            for c in C:
                score = 0
                svm = SVC(
                    C=c,
                    kernel=k,
                    gamma=g,
                    probability=True,
                    max_iter=10000,
                    random_state=42
                )
                for i in range(5):
                    # print(g, c, k, i)
                    trainX1 = []
                    trainlabel1 = []
                    valX1 = []
                    testlabel1 = []
                    for media in range(1, 15):
                        trainsample = premediasample(
                            samplepath, "trainvalidation", i, media
                        )
                        testsample = premediasample(samplepath, "test", i, media)
                        _, trainfea, trainlabel, _, _ = premediadata(
                            trainsample, featurespath, media
                        )
                        _, testfea, testlabel, _, _ = premediadata(
                            testsample, featurespath, media
                        )
                        # scaler = StandardScaler().fit(trainfea)
                        scaler = MinMaxScaler().fit(trainfea)
                        trainX = scaler.transform(trainfea)
                        valX = scaler.transform(testfea)
                        trainX1.extend(trainX)
                        valX1.extend(valX)
                        trainlabel1.extend(trainlabel)
                        testlabel1.extend(testlabel)
                    trainX2, trainlabel2 = sklearn.utils.shuffle(
                        trainX1, trainlabel1, random_state=42
                    )
                    svm.fit(trainX2, trainlabel2)
                    stapre1 = svm.predict(valX1)
                    pro1 = svm.predict_proba(valX1)

                    score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
                # score = sklearn.metrics.recall_score(vlabel, stapre1)
                score /= 5
                if score > best_score:
                    best_score = score
                    # best_parameters = {'n_neighbors':g, 'weights':c, 'p':ne}
                    best_parameters = {"gamma": g, "C": c, "kernel": k}
                    # stapre=stapre1
                    # pro=pro1
                # finalsvm=AdaBoostClassifier(tree.DecisionTreeClassifier(criterion=best_parameters['criterion'],random_state=42,
                #                             splitter=best_parameters['splitter'],class_weight='balanced'),
                #                             n_estimators=best_parameters['n_estimators'],learning_rate=best_parameters['learning_rate'])
    finalsvm = SVC(
        kernel=best_parameters["kernel"],
        gamma=best_parameters["gamma"],
        C=best_parameters["C"],
        random_state=42,
        probability=True,
        max_iter=10000,
        class_weight="balanced",
    )
    # finalsvm=KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'], p=best_parameters['p'])
    # finalsvm=Pipeline([('nca', nca), ('knn', finalsv)])
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        # pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def logicregressiontiaocan(C):
    best_score = 0
    for c in C:
        # for s in S:
        #     for p in P:
        score = 0
        svm = LogisticRegression(C=c, random_state=42, class_weight="balanced")
        for i in range(5):
            # print(g, c, k, i)
            trainX1 = []
            trainlabel1 = []
            valX1 = []
            testlabel1 = []
            for media in range(1, 15):
                trainsample = premediasample(samplepath, "trainvalidation", i, media)
                testsample = premediasample(samplepath, "test", i, media)
                _, trainfea, trainlabel, _, _ = premediadata(
                    trainsample, featurespath, media
                )
                _, testfea, testlabel, _, _ = premediadata(
                    testsample, featurespath, media
                )
                # scaler = StandardScaler().fit(trainfea)
                scaler = MinMaxScaler().fit(trainfea)
                trainX = scaler.transform(trainfea)
                valX = scaler.transform(testfea)
                trainX1.extend(trainX)
                valX1.extend(valX)
                trainlabel1.extend(trainlabel)
                testlabel1.extend(testlabel)
            trainX2, trainlabel2 = sklearn.utils.shuffle(
                trainX1, trainlabel1, random_state=42
            )
            svm.fit(trainX2, trainlabel2)
            stapre1 = svm.predict(valX1)
            pro1 = svm.predict_proba(valX1)

            score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
        # score = sklearn.metrics.recall_score(vlabel, stapre1)
        score /= 5
        if score > best_score:
            best_score = score
            best_parameters = {"C": c}

    finalsvm = LogisticRegression(
        C=best_parameters["C"], random_state=42, class_weight="balanced"
    )
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        # pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def LDAtiaocan(S, Sh):
    best_score = 0
    for s in S:
        if s == "svd":
            sh = None
            score = 0
            svm = LinearDiscriminantAnalysis(solver=s, shrinkage=sh)
            for i in range(5):
                # print(g, c, k, i)
                trainX1 = []
                trainlabel1 = []
                valX1 = []
                testlabel1 = []
                for media in range(1, 15):
                    trainsample = premediasample(
                        samplepath, "trainvalidation", i, media
                    )
                    testsample = premediasample(samplepath, "test", i, media)
                    _, trainfea, trainlabel, _, _ = premediadata(
                        trainsample, featurespath, media
                    )
                    _, testfea, testlabel, _, _ = premediadata(
                        testsample, featurespath, media
                    )
                    # scaler = StandardScaler().fit(trainfea)
                    scaler = MinMaxScaler().fit(trainfea)
                    trainX = scaler.transform(trainfea)
                    valX = scaler.transform(testfea)
                    trainX1.extend(trainX)
                    valX1.extend(valX)
                    trainlabel1.extend(trainlabel)
                    testlabel1.extend(testlabel)
                trainX2, trainlabel2 = sklearn.utils.shuffle(
                    trainX1, trainlabel1, random_state=42
                )
                svm.fit(trainX2, trainlabel2)
                stapre1 = svm.predict(valX1)
                # pro1 = svm.predict_proba(valX1)

                score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
            # score = sklearn.metrics.recall_score(vlabel, stapre1)
            score /= 5
            if score > best_score:
                best_score = score
                best_parameters = {"solver": s, "shrinkage": sh}
        else:
            for sh in Sh:
                # for c in C:
                score = 0
                svm = LinearDiscriminantAnalysis(solver=s, shrinkage=sh)
                for i in range(5):
                    # print(g, c, k, i)
                    trainX1 = []
                    trainlabel1 = []
                    valX1 = []
                    testlabel1 = []
                    for media in range(1, 15):
                        trainsample = premediasample(
                            samplepath, "trainvalidation", i, media
                        )
                        testsample = premediasample(samplepath, "test", i, media)
                        _, trainfea, trainlabel, _, _ = premediadata(
                            trainsample, featurespath, media
                        )
                        _, testfea, testlabel, _, _ = premediadata(
                            testsample, featurespath, media
                        )
                        # scaler = StandardScaler().fit(trainfea)
                        scaler = MinMaxScaler().fit(trainfea)
                        trainX = scaler.transform(trainfea)
                        valX = scaler.transform(testfea)
                        trainX1.extend(trainX)
                        valX1.extend(valX)
                        trainlabel1.extend(trainlabel)
                        testlabel1.extend(testlabel)
                    trainX2, trainlabel2 = sklearn.utils.shuffle(
                        trainX1, trainlabel1, random_state=42
                    )
                    svm.fit(trainX2, trainlabel2)
                    stapre1 = svm.predict(valX1)
                    # pro1 = svm.predict_proba(valX1)

                    score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
                # score = sklearn.metrics.recall_score(vlabel, stapre1)
                score /= 5
                if score > best_score:
                    best_score = score
                    best_parameters = {"solver": s, "shrinkage": sh}

    finalsvm = LinearDiscriminantAnalysis(
        solver=best_parameters["solver"], shrinkage=best_parameters["shrinkage"]
    )
    result = []
    for i in range(5):
        # print(g, c, k, i)
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        # pro1 = svm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def knntiaocan(N, W, P, A):
    best_score = 0
    for n in N:
        for w in W:
            for a in A:
                # print([n, w,a])
                for p in P:
                    score = 0
                    svm = KNeighborsClassifier(
                        n_neighbors=n, weights=w, p=p, algorithm=a
                    )
                    for i in range(5):
                        # print(g, c, k, i)
                        trainX1 = []
                        trainlabel1 = []
                        valX1 = []
                        testlabel1 = []
                        for media in range(1, 15):
                            trainsample = premediasample(
                                samplepath, "trainvalidation", i, media
                            )
                            testsample = premediasample(samplepath, "test", i, media)
                            _, trainfea, trainlabel, _, _ = premediadata(
                                trainsample, featurespath, media
                            )
                            _, testfea, testlabel, _, _ = premediadata(
                                testsample, featurespath, media
                            )
                            # scaler = StandardScaler().fit(trainfea)
                            scaler = MinMaxScaler().fit(trainfea)
                            trainX = scaler.transform(trainfea)
                            valX = scaler.transform(testfea)
                            trainX1.extend(trainX)
                            valX1.extend(valX)
                            trainlabel1.extend(trainlabel)
                            testlabel1.extend(testlabel)
                        trainX2, trainlabel2 = sklearn.utils.shuffle(
                            trainX1, trainlabel1, random_state=42
                        )
                        svm.fit(trainX2, trainlabel2)
                        stapre1 = svm.predict(valX1)
                        pro1 = svm.predict_proba(valX1)

                        score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
                    # score = sklearn.metrics.recall_score(vlabel, stapre1)
                    score /= 5
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            "n_neighbors": n,
                            "weights": w,
                            "p": p,
                            "algorithm": a,
                        }
    finalsvm = KNeighborsClassifier(
        n_neighbors=best_parameters["n_neighbors"],
        weights=best_parameters["weights"],
        p=best_parameters["p"],
        algorithm=best_parameters["algorithm"],
    )
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def NCtiaocan(C):
    best_score = 0
    for c in C:
        # print([g, c])
        svm = NearestCentroid(shrink_threshold=c)
        score = 0
        for i in range(5):
            # print(g, c, k, i)
            trainX1 = []
            trainlabel1 = []
            valX1 = []
            testlabel1 = []
            for media in range(1, 15):
                trainsample = premediasample(samplepath, "trainvalidation", i, media)
                testsample = premediasample(samplepath, "test", i, media)
                _, trainfea, trainlabel, _, _ = premediadata(
                    trainsample, featurespath, media
                )
                _, testfea, testlabel, _, _ = premediadata(
                    testsample, featurespath, media
                )
                # scaler = StandardScaler().fit(trainfea)
                scaler = MinMaxScaler().fit(trainfea)
                trainX = scaler.transform(trainfea)
                valX = scaler.transform(testfea)
                trainX1.extend(trainX)
                valX1.extend(valX)
                trainlabel1.extend(trainlabel)
                testlabel1.extend(testlabel)
            trainX2, trainlabel2 = sklearn.utils.shuffle(
                trainX1, trainlabel1, random_state=42
            )
            svm.fit(trainX2, trainlabel2)
            stapre1 = svm.predict(valX1)
            pro1 = svm.predict_proba(valX1)

            score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
        # score = sklearn.metrics.recall_score(vlabel, stapre1)
        score /= 5
        if score > best_score:
            best_score = score
            best_parameters = {"shrink": c}

    finalsvm = NearestCentroid(shrink_threshold=best_parameters["shrink"])
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def BYStiaocan(C):
    best_score = 0
    for c in C:
        # for g in gamma:
        # print([g, c])
        svm = CategoricalNB(alpha=c)
        score = 0
        for i in range(5):
            # print(g, c, k, i)
            trainX1 = []
            trainlabel1 = []
            valX1 = []
            testlabel1 = []
            for media in range(1, 15):
                trainsample = premediasample(samplepath, "trainvalidation", i, media)
                testsample = premediasample(samplepath, "test", i, media)
                _, trainfea, trainlabel, _, _ = premediadata(
                    trainsample, featurespath, media
                )
                _, testfea, testlabel, _, _ = premediadata(
                    testsample, featurespath, media
                )
                # scaler = StandardScaler().fit(trainfea)
                scaler = MinMaxScaler().fit(trainfea)
                trainX = scaler.transform(trainfea)
                valX = scaler.transform(testfea)
                trainX1.extend(trainX)
                valX1.extend(valX)
                trainlabel1.extend(trainlabel)
                testlabel1.extend(testlabel)
            trainX2, trainlabel2 = sklearn.utils.shuffle(
                trainX1, trainlabel1, random_state=42
            )
            svm.fit(trainX2, trainlabel2)
            stapre1 = svm.predict(valX1)
            pro1 = svm.predict_proba(valX1)

            score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
        # score = sklearn.metrics.recall_score(vlabel, stapre1)
        score /= 5
        if score > best_score:
            best_score = score
            best_parameters = {"alpha": c}
    finalsvm = CategoricalNB(best_parameters["alpha"])
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def DTreetiaocan(N, W, P, A):
    best_score = 0
    for n in N:
        for w in W:
            for a in A:
                # print([n, w,a])
                for p in P:
                    svm = tree.DecisionTreeClassifier(
                        criterion=n,
                        random_state=42,
                        splitter=w,
                        max_depth=a,
                        min_samples_split=p,
                        class_weight="balanced",
                    )
                    score = 0
                    for i in range(5):
                        # print(g, c, k, i)
                        trainX1 = []
                        trainlabel1 = []
                        valX1 = []
                        testlabel1 = []
                        for media in range(1, 15):
                            trainsample = premediasample(
                                samplepath, "trainvalidation", i, media
                            )
                            testsample = premediasample(samplepath, "test", i, media)
                            _, trainfea, trainlabel, _, _ = premediadata(
                                trainsample, featurespath, media
                            )
                            _, testfea, testlabel, _, _ = premediadata(
                                testsample, featurespath, media
                            )
                            # scaler = StandardScaler().fit(trainfea)
                            scaler = MinMaxScaler().fit(trainfea)
                            trainX = scaler.transform(trainfea)
                            valX = scaler.transform(testfea)
                            trainX1.extend(trainX)
                            valX1.extend(valX)
                            trainlabel1.extend(trainlabel)
                            testlabel1.extend(testlabel)
                        trainX2, trainlabel2 = sklearn.utils.shuffle(
                            trainX1, trainlabel1, random_state=42
                        )
                        svm.fit(trainX2, trainlabel2)
                        stapre1 = svm.predict(valX1)
                        # pro1 = svm.predict_proba(valX1)

                        score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
                    # score = sklearn.metrics.recall_score(vlabel, stapre1)
                    score /= 5
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            "criterion": n,
                            "splitter": w,
                            "max_depth": a,
                            "min_samples_split": p,
                        }
    finalsvm = tree.DecisionTreeClassifier(
        criterion=best_parameters["criterion"],
        random_state=42,
        splitter=best_parameters["splitter"],
        max_depth=best_parameters["max_depth"],
        min_samples_split=best_parameters["min_samples_split"],
        class_weight="balanced",
    )
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        # pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def RFtiaocan(N, W, P):
    best_score = 0
    for n in N:
        for w in W:
            # print([n, w,a])
            for p in P:
                svm = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=w,
                    max_features=p,
                    random_state=42,
                    class_weight="balanced",
                )
                score = 0
                for i in range(5):
                    # print(g, c, k, i)
                    trainX1 = []
                    trainlabel1 = []
                    valX1 = []
                    testlabel1 = []
                    for media in range(1, 15):
                        trainsample = premediasample(
                            samplepath, "trainvalidation", i, media
                        )
                        testsample = premediasample(samplepath, "test", i, media)
                        _, trainfea, trainlabel, _, _ = premediadata(
                            trainsample, featurespath, media
                        )
                        _, testfea, testlabel, _, _ = premediadata(
                            testsample, featurespath, media
                        )
                        # scaler = StandardScaler().fit(trainfea)
                        scaler = MinMaxScaler().fit(trainfea)
                        trainX = scaler.transform(trainfea)
                        valX = scaler.transform(testfea)
                        trainX1.extend(trainX)
                        valX1.extend(valX)
                        trainlabel1.extend(trainlabel)
                        testlabel1.extend(testlabel)
                    trainX2, trainlabel2 = sklearn.utils.shuffle(
                        trainX1, trainlabel1, random_state=42
                    )
                    svm.fit(trainX2, trainlabel2)
                    stapre1 = svm.predict(valX1)
                    pro1 = svm.predict_proba(valX1)

                    score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
                # score = sklearn.metrics.recall_score(vlabel, stapre1)
                score /= 5
                if score > best_score:
                    best_score = score
                    best_parameters = {
                        "n_estimators": n,
                        "max_depth": w,
                        "max_features": p,
                    }
    finalsvm = RandomForestClassifier(
        n_estimators=best_parameters["n_estimators"],
        max_depth=best_parameters["max_depth"],
        max_features=best_parameters["max_features"],
        random_state=42,
        class_weight="balanced",
    )
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def Adaboosttiaocan(NE, LR, N, W, P, A):
    best_score = 0
    for n in N:
        for w in W:
            # print([n, w,a])
            for p in P:
                for a in A:
                    for ne in NE:
                        for lr in LR:
                            svm = AdaBoostClassifier(
                                tree.DecisionTreeClassifier(
                                    criterion=n,
                                    random_state=42,
                                    splitter=w,
                                    max_depth=a,
                                    min_samples_split=p,
                                    class_weight="balanced",
                                ),
                                n_estimators=ne,
                                learning_rate=lr,
                            )
                            score = 0
                            for i in range(5):
                                # print(g, c, k, i)
                                trainX1 = []
                                trainlabel1 = []
                                valX1 = []
                                testlabel1 = []
                                for media in range(1, 15):
                                    trainsample = premediasample(
                                        samplepath, "trainvalidation", i, media
                                    )
                                    testsample = premediasample(
                                        samplepath, "test", i, media
                                    )
                                    _, trainfea, trainlabel, _, _ = premediadata(
                                        trainsample, featurespath, media
                                    )
                                    _, testfea, testlabel, _, _ = premediadata(
                                        testsample, featurespath, media
                                    )
                                    # scaler = StandardScaler().fit(trainfea)
                                    scaler = MinMaxScaler().fit(trainfea)
                                    trainX = scaler.transform(trainfea)
                                    valX = scaler.transform(testfea)
                                    trainX1.extend(trainX)
                                    valX1.extend(valX)
                                    trainlabel1.extend(trainlabel)
                                    testlabel1.extend(testlabel)
                                trainX2, trainlabel2 = sklearn.utils.shuffle(
                                    trainX1, trainlabel1, random_state=42
                                )
                                svm.fit(trainX2, trainlabel2)
                                stapre1 = svm.predict(valX1)
                                pro1 = svm.predict_proba(valX1)

                                score += sklearn.metrics.accuracy_score(
                                    testlabel1, stapre1
                                )
                            # score = sklearn.metrics.recall_score(vlabel, stapre1)
                            score /= 5
                            if score > best_score:
                                best_score = score
                                best_parameters = {
                                    "criterion": n,
                                    "splitter": w,
                                    "max_depth": a,
                                    "min_samples_split": p,
                                    "n_estimators": ne,
                                    "learning_rate": lr,
                                }
    finalsvm = AdaBoostClassifier(
        tree.DecisionTreeClassifier(
            criterion=best_parameters["criterion"],
            random_state=42,
            splitter=best_parameters["splitter"],
            max_depth=best_parameters["max_depth"],
            min_samples_split=best_parameters["min_samples_split"],
            class_weight="balanced",
        ),
        n_estimators=best_parameters["n_estimators"],
        learning_rate=best_parameters["learning_rate"],
    )
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            _, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            _, testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc), 4)} +/- {np.round(np.std(acc), 4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def tiaocan5cv():
    # scaler=StandardScaler().fit(tfeatures)
    # trainX=scaler.transform(tfeatures)
    # valX=scaler.transform(vfeatures)
    C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    kernel = ["rbf", "poly", "sigmoid"]
    # C = [100]
    # gamma = [0.1]
    # kernel = ['poly']
    # C = [0.1,0.2,0.3,0.4,0.5,0.09,0.08,0.07,0.06,0.05]
    # gamma = [1,2,3,4,5,0.9,0.8,0.7,0.6,0.5]

    # gamma = ['entropy', 'gini']
    # C = ['best', 'random']
    # A = [3, 4, 5, 6, 7]
    # kernel = [5, 6, 7, 8, 9, 10]
    # NE = [50, 30, 70, 90, 160, 200, 250, 300]
    # LR = [0.1,  0.3, 0.5, 0.7, 0.9, 1]

    # gamma = [2,3,4,5,6,7,8,9,10]
    # C = ['uniform', 'distance']
    # A = [3, 4, 5, 6, 7]
    # kernel = [5, 6, 7, 8, 9, 10]
    # NE = [1,2,3,4,5]
    # LR = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    best_score = 0
    # nca = NeighborhoodComponentsAnalysis(random_state=42)
    for g in gamma:
        for c in C:
            for k in kernel:
                # for ne in NE:
                # for lr in LR:
                # print([g, c])
                svm = SVC(
                    kernel=k,
                    gamma=g,
                    C=c,
                    random_state=42,
                    probability=True,
                    max_iter=10000,
                    class_weight="balanced",
                )
                #         svm=AdaBoostClassifier(tree.DecisionTreeClassifier(criterion=g,random_state=42,splitter=c,class_weight='balanced'),
                #                                n_estimators=ne,learning_rate=lr)
                #     svm=KNeighborsClassifier(n_neighbors=g,weights=c,p=ne)
                #     svm = Pipeline([('nca', nca), ('knn', sv)])
                score = 0
                for i in range(5):
                    print(g, c, k, i)
                    trainX1 = []
                    trainlabel1 = []
                    valX1 = []
                    testlabel1 = []
                    for media in range(1, 15):
                        trainsample = premediasample(
                            samplepath, "trainvalidation", i, media
                        )
                        testsample = premediasample(samplepath, "test", i, media)
                        trainfea, trainlabel, _, _ = premediadata(
                            trainsample, featurespath, media
                        )
                        testfea, testlabel, _, _ = premediadata(
                            testsample, featurespath, media
                        )
                        # scaler = StandardScaler().fit(trainfea)
                        scaler = MinMaxScaler().fit(trainfea)
                        trainX = scaler.transform(trainfea)
                        valX = scaler.transform(testfea)
                        trainX1.extend(trainX)
                        valX1.extend(valX)
                        trainlabel1.extend(trainlabel)
                        testlabel1.extend(testlabel)
                    trainX2, trainlabel2 = sklearn.utils.shuffle(
                        trainX1, trainlabel1, random_state=42
                    )
                    svm.fit(trainX2, trainlabel2)
                    stapre1 = svm.predict(valX1)
                    pro1 = svm.predict_proba(valX1)

                    score += sklearn.metrics.accuracy_score(testlabel1, stapre1)
                # score = sklearn.metrics.recall_score(vlabel, stapre1)
                score /= 5
                if score > best_score:
                    best_score = score
                    # best_parameters = {'n_neighbors':g, 'weights':c, 'p':ne}
                    best_parameters = {"gamma": g, "C": c, "kernel": k}
                    # stapre=stapre1
                    # pro=pro1
    # finalsvm=AdaBoostClassifier(tree.DecisionTreeClassifier(criterion=best_parameters['criterion'],random_state=42,
    #                             splitter=best_parameters['splitter'],class_weight='balanced'),
    #                             n_estimators=best_parameters['n_estimators'],learning_rate=best_parameters['learning_rate'])
    finalsvm = SVC(
        kernel=best_parameters["kernel"],
        gamma=best_parameters["gamma"],
        C=best_parameters["C"],
        random_state=42,
        probability=True,
        max_iter=10000,
        class_weight="balanced",
    )
    # finalsvm=KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'], p=best_parameters['p'])
    # finalsvm=Pipeline([('nca', nca), ('knn', finalsv)])
    result = []
    for i in range(5):
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            trainfea, trainlabel, _, _ = premediadata(trainsample, featurespath, media)
            testfea, testlabel, _, _ = premediadata(testsample, featurespath, media)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)

        result.append(metrics(testlabel1, stapre1, i))
    print(best_parameters)
    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc),4)} +/- {np.round(np.std(acc),4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def mediatiaocan5cv():
    for media in range(1, 15):
        C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        kernel = ["rbf", "poly", "sigmoid"]
        # C = [0.1,0.2,0.3,0.4,0.5,0.09,0.08,0.07,0.06,0.05]
        # gamma = [1,2,3,4,5,0.9,0.8,0.7,0.6,0.5]

        # gamma = ['entropy', 'gini']
        # C = ['best', 'random']
        # A = [3, 4, 5, 6, 7]
        # kernel = [5, 6, 7, 8, 9, 10]
        # NE = [50, 30, 70, 90, 160, 200, 250, 300]
        # LR = [0.1,  0.3, 0.5, 0.7, 0.9, 1]

        # gamma = [2,3,4,5,6,7,8,9,10]
        # C = ['uniform', 'distance']
        # A = [3, 4, 5, 6, 7]
        # kernel = [5, 6, 7, 8, 9, 10]
        # NE = [1,2,3,4,5]
        # LR = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        best_score = 0
        nca = NeighborhoodComponentsAnalysis(random_state=42)
        for g in gamma:
            for c in C:
                for k in kernel:
                    # for ne in NE:
                    # for lr in LR:
                    # print([g, c])
                    svm = SVC(
                        kernel=k,
                        gamma=g,
                        C=c,
                        random_state=42,
                        probability=True,
                        max_iter=10000,
                        class_weight="balanced",
                    )
                    #         svm=AdaBoostClassifier(tree.DecisionTreeClassifier(criterion=g,random_state=42,splitter=c,class_weight='balanced'),
                    #                                n_estimators=ne,learning_rate=lr)
                    #     svm=KNeighborsClassifier(n_neighbors=g,weights=c,p=ne)
                    #     svm = Pipeline([('nca', nca), ('knn', sv)])
                    score = 0
                    for i in range(5):
                        # print(g,c,k,i)
                        trainsample = premediasample(
                            samplepath, "trainvalidation", i, media
                        )
                        testsample = premediasample(samplepath, "test", i, media)
                        trainfea, trainlabel, _, _ = premediadata(
                            trainsample, featurespath, media
                        )
                        testfea, testlabel, _, _ = premediadata(
                            testsample, featurespath, media
                        )
                        # scaler = StandardScaler().fit(trainfea)
                        scaler = MinMaxScaler().fit(trainfea)
                        trainX = scaler.transform(trainfea)
                        valX = scaler.transform(testfea)
                        svm.fit(trainX, trainlabel)
                        stapre1 = svm.predict(valX)
                        pro1 = svm.predict_proba(valX)

                        score += sklearn.metrics.accuracy_score(testlabel, stapre1)
                    # score = sklearn.metrics.recall_score(vlabel, stapre1)
                    score /= 5
                    if score > best_score:
                        best_score = score
                        # best_parameters = {'n_neighbors':g, 'weights':c, 'p':ne}
                        best_parameters = {"gamma": g, "C": c, "kernel": k}
                        # stapre=stapre1
                        # pro=pro1
        # finalsvm=AdaBoostClassifier(tree.DecisionTreeClassifier(criterion=best_parameters['criterion'],random_state=42,
        #                             splitter=best_parameters['splitter'],class_weight='balanced'),
        #                             n_estimators=best_parameters['n_estimators'],learning_rate=best_parameters['learning_rate'])
        finalsvm = SVC(
            kernel=best_parameters["kernel"],
            gamma=best_parameters["gamma"],
            C=best_parameters["C"],
            random_state=42,
            probability=True,
            max_iter=10000,
            class_weight="balanced",
        )
        # finalsvm=KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'], p=best_parameters['p'])
        # finalsvm=Pipeline([('nca', nca), ('knn', finalsv)])
        result = []
        for i in range(5):
            trainsample = presample(samplepath, "trainvalidation", i)
            testsample = presample(samplepath, "test", i)
            trainfea, trainlabel, _, _ = predata(trainsample, featurespath)
            testfea, testlabel, _, _ = predata(testsample, featurespath)
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            finalsvm.fit(trainX, trainlabel)
            stapre1 = finalsvm.predict(valX)
            pro1 = finalsvm.predict_proba(valX)
            result.append(metrics(testlabel, stapre1, pro1, i))
        print("media" + str(media))
        print(best_parameters)
        print(result)
        acc = [re[1] for re in result]
        pre = [re[2] for re in result]
        recall = [re[3] for re in result]
        f1 = [re[4] for re in result]
        speci = [re[5] for re in result]
        auc = [re[6] for re in result]

        print(f"Acc: {np.round(np.mean(acc),4)} +/- {np.round(np.std(acc),4)}")
        print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
        print(
            f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}"
        )
        print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
        print(
            f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
        )
        print(f"Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}")


def soft(participant, sample, samplelabel, samplepre, yz):
    prelabel = []
    subject_pro = []
    for i in range(len(participant) - 1):
        record0 = []
        record1 = []
        e = 0
        for j in range(len(sample)):
            if sample[j][:-1] == participant[i]:
                e = 1
                record0.append(samplepre[j][0])
                record1.append(samplepre[j][1])

        if np.mean(record1) >= yz:
            prelabel.append(1)
        else:
            prelabel.append(0)
        subject_pro.append(np.mean(record1))

    return prelabel,subject_pro


def hard(participant, sample, samplelabel, samplepre):
    prelabel = []
    for i in range(len(participant) - 1):
        record0 = []
        e = 0
        for j in range(len(sample)):
            if sample[j][:-1] == participant[i]:
                record0.append(samplelabel[j])
        if record0.count(1) >= record0.count(0):
            prelabel.append(1)
        else:
            prelabel.append(0)
    return prelabel


def vote(samplepath, rou, sample, samplelabel, samplepre, yz):
    fo = open(f"{samplepath}/participant14.txt", "r")
    participant = []
    myFile = fo.read()
    myRecords = myFile.split("\n")
    for y in range(0, len(myRecords) - 1):
        participant.append(myRecords[y].split("\t"))  # 把表格数据存入

    label = []
    for i in range(len(participant[rou]) - 1):
        if participant[rou][i][0] == "n":
            label.append(0)
        else:
            label.append(1)

    # prelabel = hard(participant[rou], sample, samplelabel, samplepre)
    prelabel, subject_pro  = soft(participant[rou], sample, samplelabel, samplepre, yz)

    # if len(wu)!=0:
    #     counter=0
    #     for index in wu:
    #         index = index - counter
    #         participant[rou].pop(index)
    #         label.pop(index)
    #         counter += 1

    return label, prelabel,subject_pro


def vote5cv():
    gamma = 0.1
    c = 10
    k = "poly"

    n_neighbors = 8
    weights = "distance"
    p = 3
    algorithm = "auto"

    result = []
    for i in range(5):
        finalsvm = SVC(
            gamma=gamma,
            C=c,
            kernel=k,
            random_state=42,
            probability=True,
            class_weight="balanced",
        )
        # finalsvm=LogisticRegression(C=c, random_state=42)
        # finalsvm=KNeighborsClassifier(n_neighbors= 8, weights= 'distance', p= 3, algorithm= 'auto')
        # finalsvm=RandomForestClassifier(n_estimators= 100, max_depth=10, max_features= 'sqrt')
        trainX1 = []
        trainlabel1 = []
        valX1 = []
        testlabel1 = []
        testsample1 = []
        for media in range(1, 15):
            trainsample = premediasample(samplepath, "trainvalidation", i, media)
            testsample = premediasample(samplepath, "test", i, media)
            retrainsample, trainfea, trainlabel, _, _ = premediadata(
                trainsample, featurespath, media
            )
            retestsample, testfea, testlabel, _, _ = premediadata(
                testsample, featurespath, media
            )
            # scaler = StandardScaler().fit(trainfea)
            scaler = MinMaxScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            trainX1.extend(trainX)
            valX1.extend(valX)
            trainlabel1.extend(trainlabel)
            testlabel1.extend(testlabel)
            testsample1.extend(retestsample)
        trainX2, trainlabel2 = sklearn.utils.shuffle(
            trainX1, trainlabel1, random_state=42
        )
        finalsvm.fit(trainX2, trainlabel2)
        stapre1 = finalsvm.predict(valX1)
        pro1 = finalsvm.predict_proba(valX1)
        # print(pro1)

        plabel, ppre = vote(samplepath, i, testsample1, stapre1, pro1, 0.6)

        result.append(metrics(plabel, ppre, i))

    print(result)
    acc = [re[1] for re in result]
    pre = [re[2] for re in result]
    recall = [re[3] for re in result]
    f1 = [re[4] for re in result]
    speci = [re[5] for re in result]
    # auc = [re[6] for re in result]

    print(f"Acc: {np.round(np.mean(acc),4)} +/- {np.round(np.std(acc),4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    # print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def bestsoftvote5cv():
    # gamma = 0.1
    # c = 100
    # k = "poly"

    # c = 2302.037957709518
    # gamma =  0.0003631701222484867
    # k = 'sigmoid'

    gamma = 7.112519372117792
    c = 37.48928620282189
    k = "poly"

    bestacc = 0
    result1 = []
    bestyz = 50
    for y in range(40,60): #原来（50,51）
        result = []
        segment_result = []
        res = 0
        yz = y * 0.01
        for i in range(5):
            finalsvm = SVC(
                gamma=gamma,
                C=c,
                kernel=k,
                random_state=42,
                max_iter=1000,
                class_weight="balanced",
                probability=True,
            )
            trainX1 = []
            trainlabel1 = []
            valX1 = []
            testlabel1 = []
            testsample1 = []
            for media in range(1, 15):
                trainsample = premediasample(samplepath, "trainvalidation", i, media)
                testsample = premediasample(samplepath, "test", i, media)
                retrainsample, trainfea, trainlabel, _, _ = premediadata(
                    trainsample, featurespath, media
                )
                retestsample, testfea, testlabel, _, _ = premediadata(
                    testsample, featurespath, media
                )
                # scaler = StandardScaler().fit(trainfea)
                scaler = MinMaxScaler().fit(trainfea)
                trainX = scaler.transform(trainfea)
                valX = scaler.transform(testfea)
                trainX1.extend(trainX)
                valX1.extend(valX)
                trainlabel1.extend(trainlabel)
                testlabel1.extend(testlabel)
                testsample1.extend(retestsample)
            trainX2, trainlabel2 = sklearn.utils.shuffle(
                trainX1, trainlabel1, random_state=42
            )
            finalsvm.fit(trainX2, trainlabel2)
            stapre1 = finalsvm.predict(valX1)
            pro1 = finalsvm.predict_proba(valX1)

            segment_result.append(metrics(testlabel1, stapre1, i, pro1[:,1]))
            plabel, ppre, subject_pro = vote(samplepath, i, testsample1, stapre1, pro1, yz)
            result.append(metrics(plabel, ppre, i, subject_pro))
            res += metrics(plabel, ppre, i,subject_pro)[1]
        if res / 5 > bestacc:
            best_segment_result = segment_result
            result1 = result
            bestyz = yz
            bestacc = res / 5
            with open(f"svm_model.pkl", "wb") as file:
                pickle.dump(finalsvm, file)

    print("Segment:")
    print(best_segment_result)
    acc = [re[1] for re in best_segment_result]
    pre = [re[2] for re in best_segment_result]
    recall = [re[3] for re in best_segment_result]
    f1 = [re[4] for re in best_segment_result]
    speci = [re[5] for re in best_segment_result]
    auc = [re[6] for re in best_segment_result]
    print(f"")
    print(f"Acc: {np.round(np.mean(acc),4)} +/- {np.round(np.std(acc),4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')

    print("Subject:")
    print(bestyz)
    print(result1)
    acc = [re[1] for re in result1]
    pre = [re[2] for re in result1]
    recall = [re[3] for re in result1]
    f1 = [re[4] for re in result1]
    speci = [re[5] for re in result1]
    auc = [re[6] for re in result]
    print(f"")
    print(f"Acc: {np.round(np.mean(acc),4)} +/- {np.round(np.std(acc),4)}")
    print(f"Precision: {np.round(np.mean(pre), 4)} +/- {np.round(np.std(pre), 4)}")
    print(f"Recall: {np.round(np.mean(recall), 4)} +/- {np.round(np.std(recall), 4)}")
    print(f"F1: {np.round(np.mean(f1), 4)} +/- {np.round(np.std(f1), 4)}")
    print(
        f"Specificity: {np.round(np.mean(speci), 4)} +/- {np.round(np.std(speci), 4)}"
    )
    print(f'Auc: {np.round(np.mean(auc), 4)} +/- {np.round(np.std(auc), 4)}')


def printtxt(sample, label, stapre, pro, rou, resultpath):
    fo = open(f"{resultpath}/{rou}.txt", "w")
    for i in range(len(stapre)):
        fo.write(str(sample[i][0]))
        fo.write("\t")
        fo.write(str(sample[i][1]))
        fo.write("\t")
        fo.write(str(label[i]))
        fo.write("\t")
        fo.write(str(stapre[i]))
        fo.write("\t")
        fo.write(str(pro[i][0]))
        fo.write("\t")
        fo.write(str(pro[i][1]))
        fo.write("\n")
    fo.close()


def metrics(testy, stapre1, j, pro1):
    accuracy1 = sklearn.metrics.accuracy_score(testy, stapre1)
    precision1 = sklearn.metrics.precision_score(testy, stapre1)
    recall1 = sklearn.metrics.recall_score(testy, stapre1)
    f11 = sklearn.metrics.f1_score(testy, stapre1)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(testy, stapre1).ravel()
    speci = tn / (tn + fp)
    auc1 = sklearn.metrics.roc_auc_score(testy, pro1)

    return [
        "round" + str(j + 1),
        np.round(accuracy1, 4),
        np.round(precision1, 4),
        np.round(recall1, 4),
        np.round(f11, 4),
        np.round(speci, 4),
        np.round(auc1,4)
    ]


def cv5(samplepath, featurespath, resultpath, gamma, C, kernel, A, NE, LR):
    result = []
    for i in range(5):
        trains1 = presample(samplepath, "trainvalidation", i)
        tests1 = presample(samplepath, "test", i)
        trainf1, trainl1, asd, td = predata(trains1, featurespath)
        testf1, testl1, _, _ = predata(tests1, featurespath)
        best_parameters, stapre1, pro1 = Adaboosttiaocan(
            NE[i],
            LR[i],
            gamma[i],
            C[i],
            kernel[i],
            A[i],
            trainf1,
            testf1,
            trainl1,
            testl1,
            asd,
            td,
        )
        # printtxt(tests1,testl1,stapre1,pro1,i,resultpath)
        result.append(metrics(testl1, stapre1, pro1, i))
    print(result)

    # for i in range(1,5):
    #     trainsample=presample(samplepath,'trainvalidation',i)
    #     testsample = presample(samplepath,'test', i)
    #     trainfea,trainlable=predata(trainsample,featurespath)
    #     testfea, testlable=predata(testsample,featurespath)
    #     scaler = StandardScaler().fit(trainfea)
    #     trainX = scaler.transform(trainfea)
    #     valX = scaler.transform(testfea)
    #     finalsvm = SVC(kernel='rbf', gamma=best_parameters['gamma'], C=best_parameters['C'],
    #                    random_state=42, probability=True)
    #     finalsvm.fit(trainX,trainlable)
    #     stapre = finalsvm.predict(valX)
    #     pro = finalsvm.predict_proba(valX)
    #     printtxt(stapre, pro, i, resultpath)
    #     result.append(metrics(testlable, stapre, pro, i))

    fo = open(f"{resultpath}/result.txt", "w")
    fo.write(
        "round"
        + "\t"
        + "acc"
        + "\t"
        + "precision"
        + "\t"
        + "recall"
        + "\t"
        + "f1"
        + "\t"
        + "specificity"
        + "\t"
        + "auc"
        + "\n"
    )
    for i in range(len(result)):
        for j in range(len(result[i])):
            fo.write(str(result[i][j]) + "\t")
        fo.write("\n")
    fo.close()


def cccv5(samplepath, featurespath, resultpath, gamma, C, kernel, A, NE, LR, i):
    result = []
    trains1 = presample(samplepath, "trainvalidation", i)
    tests1 = presample(samplepath, "test", i)
    trainf1, trainl1, asd, td = predata(trains1, featurespath)
    testf1, testl1, _, _ = predata(tests1, featurespath)
    # best_parameters, stapre1, pro1 = Adaboosttiaocan(
    #     NE[i],
    #     LR[i],
    #     gamma[i],
    #     C[i],
    #     kernel[i],
    #     A[i],
    #     trainf1,
    #     testf1,
    #     trainl1,
    #     testl1,
    #     asd,
    #     td,
    # )
    # printtxt(tests1,testl1,stapre1,pro1,i,resultpath)
    # result.append(metrics(testl1, stapre1, pro1, i))
    # print(result)

    for j in range(0, 5):
        if j != i:
            trainsample = presample(samplepath, "trainvalidation", j)
            testsample = presample(samplepath, "test", j)
            trainfea, trainlable = predata(trainsample, featurespath)
            testfea, testlable = predata(testsample, featurespath)
            scaler = StandardScaler().fit(trainfea)
            trainX = scaler.transform(trainfea)
            valX = scaler.transform(testfea)
            finalsvm = SVC(
                kernel="rbf",
                gamma=best_parameters["gamma"],
                C=best_parameters["C"],
                random_state=42,
                probability=True,
            )
            finalsvm.fit(trainX, trainlable)
            stapre = finalsvm.predict(valX)
            pro = finalsvm.predict_proba(valX)
            # printtxt(stapre, pro, j, resultpath)
            result.append(metrics(testlable, stapre, pro, j))

    fo = open(f"{resultpath}/result.txt", "w")
    fo.write(
        "round"
        + "\t"
        + "acc"
        + "\t"
        + "precision"
        + "\t"
        + "recall"
        + "\t"
        + "f1"
        + "\t"
        + "specificity"
        + "\t"
        + "auc"
        + "\n"
    )
    for j in range(len(result)):
        for j in range(len(result[j])):
            fo.write(str(result[j][j]) + "\t")
        fo.write("\n")
    fo.close()


# featurespath = "stafeatures/stafeatureswithmeand"
featurespath = "stafeatures/aoiwithduration"

samplepath = "../sample/20220817wos4"
resultpath = "result/segment"

if not os.path.exists(resultpath):
    os.makedirs(resultpath)

print("SVM new")
# kernel = ["rbf", "poly", "sigmoid"]
# C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
# gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
C = [ 100]
gamma = [0.1]
kernel = ['poly']
# kernel = ["rbf", "poly", "sigmoid"]
# C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]
# gamma = [ 1e-3, 0.01, 0.1, 1, 10, 100]
# svmtiaocan(C, gamma, kernel)


print("logicregression new")
# solver=['liblinear', 'newton-cg','lbfgs', 'sag' , 'saga']
# penalty=['l1','l2', 'elasticnet', None]
Cs = [1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
# logicregressiontiaocan(Cs)

# print('LDA')
# solver1=['svd','lsqr','eigen']
# shrink=['auto',None]
# # components=[3,5,7,9,10,11,13]
# LDAtiaocan(solver1,shrink)

print("knn new")
Nestimater = [3, 4, 5, 6, 7, 8, 9, 10]
weights = ["uniform", "distance"]
P = [1, 2, 3, 4, 5, 6]
algorthm = ["auto", "ball_tree", "kd_tree", "brute"]
# knntiaocan(Nestimater, weights, P, algorthm)

print("DecisionTree")
criterion = ["entropy", "gini"]
splitter = ["best", "random"]
split = [2, 3, 4, 5, 6]
maxdep = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# criterion=['gini']
# splitter=['best']
# split=[5]
# maxdep=[None]
# DTreetiaocan(criterion, splitter, split, maxdep)

print("RandomForest")
nemi = [5, 10, 25, 50, 100, 200]
maxd = [10, 30, 50, 70, 90]
maxf = [None, "sqrt", "auto", "log2"]
# RFtiaocan(nemi, maxd, maxf)

# A = [3, 4, 5, 6, 7]
# NE = [1, 2, 3, 4, 5]
# LR = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
# cv5(samplepath, featurespath, resultpath, gamma, C, kernel, A, NE, LR)
# tiaocan5cv()
# vote5cv()
bestsoftvote5cv()

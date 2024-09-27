import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import sklearn
import pickle
import torch
import torch.nn as nn
import torchvision.models as mod
import xlwt
import model
from torchvision import transforms
from PIL import Image
from dataloader import pallmediaMyDataset
import train

size = {
    "media1": 234,
    "media2": 286,
    "media3": 44,
    "media4": 225,
    "media5": 186,
    "media6": 64,
    "media7": 80,
    "media8": 83,
    "media9": 152,
    "media10": 128,
    "media11": 135,
    "media12": 58,
    "media13": 75,
    "media14": 86,
}


class parameter:
    def __init__(self):
        self.lr_scheduler = "plateau"
        self.gamma = 0.5
        self.epochs = 50
        self.lr = 1e-6
        self.batch_size = 64
        self.flush_history = 0
        self.patience = 5
        self.save_model = 1
        self.log_every = 100
        self.version = "v20220626"


args = parameter()
samplepath = "../../sample/gazepoint/10"
imgpath = "../../DataProcessing/gazepointimg/fsjoint1d"

for rou in range(5):
    print("round " + str(rou))
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
        print("use gpu")
    mo = model.resnet().cuda()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize(
                [
                    0.485,
                ],
                [
                    0.229,
                ],
            ),
        ]
    )

    participants = []
    mod = os.listdir(f"v20220626/models/10gzresnet/{rou}/64_1e-06")
    mo.load_state_dict(
        torch.load(f"v20220626/models/10gzresnet/{rou}/64_1e-06/{mod[0]}")
    )
    fo = open(f"../../sample/gazepoint/10/{rou}/trainsample.txt", "r")
    myFile = fo.read()
    myRecords = myFile.split("\n")
    for y in range(0, len(myRecords) - 1):
        participants.append(myRecords[y].split("\t"))  # 把表格数据存入

    trainsample = []
    trainy = []
    if participants[0][1][0] == "n":
        trainy.append(0)
    else:
        trainy.append(1)
    trainsample.append(participants[0][1][:-1])

    for i in range(1, len(participants)):
        jl = 0
        for j in range(len(trainsample)):
            if participants[i][1][:-1] == trainsample[j]:
                jl = 0
                break
            else:
                jl += 1

        if jl == len(trainsample):
            trainsample.append(participants[i][1][:-1])
            if participants[i][1][0] == "n":
                trainy.append(0)
            else:
                trainy.append(1)
    print(trainsample)
    print(trainy)

    f1 = open(f"../../sample/gazepoint/10/{rou}/testsample.txt", "r")
    participants1 = []
    myFile1 = f1.read()
    myRecords1 = myFile1.split("\n")
    for y in range(0, len(myRecords1) - 1):
        participants1.append(myRecords1[y].split("\t"))  # 把表格数据存入

    testsample = []
    testy = []
    if participants1[0][1][0] == "n":
        testy.append(0)
    else:
        testy.append(1)
    testsample.append(participants1[0][1][:-1])

    for i in range(1, len(participants1)):
        jl = 0
        for j in range(len(testsample)):
            if participants1[i][1][:-1] == testsample[j]:
                jl = 0
                break
            else:
                jl += 1

        if jl == len(testsample):
            testsample.append(participants1[i][1][:-1])
            if participants1[i][1][0] == "n":
                testy.append(0)
            else:
                testy.append(1)
    print(testsample)
    print(testy)

    medias1 = [
        "media1",
        "media2",
        "media4",
        "media5",
        "media7",
        "media8",
        "media9",
        "media10",
        "media11",
        "media13",
    ]
    trainx = []
    testx = []

    train_dataset = pallmediaMyDataset(imgpath, samplepath, rou, "train")
    test_dataset = pallmediaMyDataset(imgpath, samplepath, rou, "test")
    val_dataset = pallmediaMyDataset(imgpath, samplepath, rou, "validation")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
    )
    n1, n2, n3, n4, trainproba, n5, _ = train.evaluate_model(mo, train_loader, device)
    n1, n2, n3, n4, testproba, n5, _ = train.evaluate_model(mo, test_loader, device)
    n1, n2, n3, n4, valproba, n5, _ = train.evaluate_model(mo, val_loader, device)

    for i in range(len(trainsample)):
        trainxi = []
        for k in range(len(medias1)):
            for o in range(1, 4):
                trainxi.append(0.5)
        trainx.append(trainxi)

    for i in range(len(trainsample)):
        for k in range(len(medias1)):
            for o in range(1, 4):
                for j in range(len(participants)):
                    if participants[j][0] == medias1[k] and participants[j][
                        1
                    ] == trainsample[i] + str(o):
                        # image1 = Image.open(
                        #     f'../../DataProcessing/gazepointimg/fsjoint1d/{participants[j][0]}/{participants[j][1]}.jpg')
                        # image1=transform(image1)
                        # image1=torch.unsqueeze(image1,0)
                        # image1 = image1.to(device)
                        #
                        # prediction= mo.forward(image1.float(),size[participants[j][0]])
                        # probas = torch.softmax(prediction, dim=1)
                        # probas=probas.cpu()
                        # probas=probas.detach().numpy()
                        # proba=probas[0][1]
                        trainx[i][int(k * 3 + o - 1)] = trainproba[j]
                        break

    for i in range(len(testsample)):
        testxi = []
        for k in range(len(medias1)):
            for o in range(1, 4):
                testxi.append(0.5)
                for j in range(len(participants1)):
                    if participants1[j][0] == medias1[k] and participants1[j][
                        1
                    ] == testsample[i] + str(o):
                        # image1 = Image.open(
                        #     f'../../DataProcessing/gazepointimg/fsjoint1d/{participants1[j][0]}/{participants1[j][1]}.jpg')
                        # image1 = transform(image1)
                        # image1 = torch.unsqueeze(image1, 0)
                        # image1 = image1.to(device)
                        #
                        # prediction = mo.forward( image1.float(),size[participants[j][0]])
                        # probas = torch.softmax(prediction, dim=1)
                        # probas = probas.cpu()
                        # probas = probas.detach().numpy()
                        # proba = probas[0][1]
                        testxi[int(k * 3 + o - 1)] = testproba[j]
        testx.append(testxi)

    f2 = open(f"../../sample/gazepoint/10/{rou}/validationsample.txt", "r")
    participants2 = []
    myFile2 = f2.read()
    myRecords2 = myFile2.split("\n")
    for y in range(0, len(myRecords2) - 1):
        participants2.append(myRecords2[y].split("\t"))  # 把表格数据存入

    vsample = []
    vy = []
    if participants2[0][1][0] == "n":
        vy.append(0)
    else:
        vy.append(1)
    vsample.append(participants2[0][1][:-1])

    for i in range(1, len(participants2)):
        jl = 0
        for j in range(len(vsample)):
            if participants2[i][1][:-1] == vsample[j]:
                jl = 0
                break
            else:
                jl += 1

        if jl == len(vsample):
            vsample.append(participants2[i][1][:-1])
            if participants2[i][1][0] == "n":
                vy.append(0)
            else:
                vy.append(1)
    print(vsample)
    print(vy)

    vx = []

    for i in range(len(vsample)):
        vxi = []
        for k in range(len(medias1)):
            for o in range(1, 4):
                vxi.append(0.5)
                for j in range(len(participants2)):
                    if participants2[j][0] == medias1[k] and participants2[j][
                        1
                    ] == vsample[i] + str(o):
                        # image1 = Image.open(
                        #     f'../../DataProcessing/gazepointimg/fsjoint1d/{participants2[j][0]}/{participants2[j][1]}.jpg')
                        # image1=transform(image1)
                        # image1=torch.unsqueeze(image1,0)
                        # image1 = image1.to(device)
                        #
                        # prediction= mo.forward(image1.float(),size[participants[j][0]])
                        # probas = torch.softmax(prediction, dim=1)
                        # probas=probas.cpu()
                        # probas=probas.detach().numpy()
                        # proba=probas[0][1]
                        vxi[int(k * 3 + o - 1)] = valproba[j]
        vx.append(vxi)

    alltrainx = trainx
    alltrainx.extend(vx)
    alltrainy = trainy
    alltrainy.extend(vy)
    print(alltrainy)

    gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    kernel = ["rbf", "linear", "poly", "sigmoid"]

    (
        bestauc,
        bestacc,
        bestpre,
        bestre,
        bestf1,
        bestsp,
        bestaucv,
        bestaccv,
        bestprev,
        bestrev,
        bestf1v,
        bestspv,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    bestg = 0
    bestc = 0
    bestk = "rbf"
    pre, prob = [], []
    foldx = []
    foldy = []
    zc = len(alltrainy) // 5
    ys = len(alltrainy) % 5
    qs = 0
    for i in range(5):
        foldxi, foldyi = [], []
        if i < ys:
            for j in range(qs, int(zc * (i + 1) + 1)):
                foldxi.append(alltrainx[j])
                foldyi.append(alltrainy[j])
        else:
            for j in range(qs, int(zc * (i + 1))):
                foldxi.append(alltrainx[j])
                foldyi.append(alltrainy[j])
        foldx.append(foldxi)
        foldy.append(foldyi)
        qs += len(foldyi)
    print(foldy)

    for g in gamma:
        for c in C:
            for k in kernel:
                acc, precision, recall, f1, specificity = [], [], [], [], []
                for i in range(5):
                    x, y = [], []
                    tx = foldx[i]
                    ty = foldy[i]
                    for j in range(5):
                        if j != i:
                            x.extend(foldx[j])
                            y.extend(foldy[j])
                    svm = SVC(
                        C=c,
                        kernel=k,
                        gamma=g,
                        random_state=42,
                        probability=True,
                        class_weight={0: 1, 1: 2},
                    )
                    # svm = SVC(C=c, kernel='rbf', gamma=g, random_state=42, probability=True)
                    stapre, pro = [], []

                    svm.fit(x, y)

                    stapre1 = svm.predict(tx)
                    stapre.append(stapre1)
                    pro1 = svm.predict_proba(tx)
                    pro.append(pro1)
                    # print(ty)
                    # print(stapre1)

                    accuracy1 = sklearn.metrics.accuracy_score(ty, stapre1)
                    precision1 = sklearn.metrics.precision_score(ty, stapre1)
                    recall1 = sklearn.metrics.recall_score(ty, stapre1)
                    f11 = sklearn.metrics.f1_score(ty, stapre1)
                    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
                        ty, stapre1
                    ).ravel()
                    speci = tn / (tn + fp)

                    acc.append(accuracy1)
                    precision.append(precision1)
                    recall.append(recall1)
                    f1.append(f11)
                    specificity.append(speci)

                accuracy1 = np.mean(acc)
                precision1 = np.mean(precision)
                recall1 = np.mean(recall)
                f11 = np.mean(f1)
                speci = np.mean(specificity)
                if accuracy1 > bestacc:
                    bestacc = accuracy1
                    bestpre = precision1
                    bestre = recall1
                    bestf1 = f11
                    bestsp = speci

                    bestc = c
                    bestg = g
                    bestk = k

    print("validation")
    print("accuracy:" + str(bestacc))
    print("precision:" + str(bestpre))
    print("recall:" + str(bestre))
    print("f1:" + str(bestf1))
    print("specificity:" + str(bestsp))

    svm = SVC(
        C=bestc,
        kernel=bestk,
        gamma=bestg,
        random_state=42,
        probability=True,
        class_weight={0: 1, 1: 2},
    )
    # svm=SVC(kernel='rbf',random_state=42,class_weight={0:1,1:2})
    svm.fit(alltrainx, alltrainy)
    testpre = svm.predict(testx)

    print("kernel: " + bestk + " gamma: " + str(bestg) + " C: " + str(bestc))
    print("accuracy:" + str(sklearn.metrics.accuracy_score(testy, testpre)))
    print("precision:" + str(sklearn.metrics.precision_score(testy, testpre)))
    print("recall:" + str(sklearn.metrics.recall_score(testy, testpre)))
    print("f1:" + str(sklearn.metrics.f1_score(testy, testpre)))
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(testy, testpre).ravel()
    speci = tn / (tn + fp)
    print("specificity:" + str(speci))

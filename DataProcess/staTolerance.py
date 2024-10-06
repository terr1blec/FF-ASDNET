import os
import xlwt
import numpy as np
import pandas as pd

class paramas():
    def __init__(self):
        self.datafile = '20210623'
        self.savefile = '1datadecolumn'

        self.savesaccadecoor = '2saccadecoor'
        self.savegazepointcoor = '2gazepointcoor'

        self.savesaccadefixation = '3saccadefixation'
        self.savefixation = '3fixation'
        self.savegazepoint = '3gazepoint'
        self.videostart = 'videostart'

        self.oksaccadefixation = '5oksaccadefixation'
        self.okfixation = '5okfixation'
        self.okgazepoint = '5okgazepoint'
        self.videostartsplit = 'videostartsplit'

        self.fixationsample = '6fixationsample'
        self.saccadefixationsample = '6saccadefixationsample'
        self.gazepointsample = '6gazepointsample'

        self.fixationimgcoor = 'fixationimg/fixationcoor'
        self.saccadefixationimgcoor = 'saccadefixationimg/fixationcoor'
        self.gazepointimgcoor = 'gazepointimg/fixationcoor'
        self.gazepointfps = 'gazepointfps'

        self.fixationfsjoint = 'fixationimg/fsjoint14ss'
        self.saccadefixationfsjoint = 'saccadefixationimg/fsjoint10ss'
        self.gazepointfsjoint = 'gazepointimg/fsjoint14ss'
        self.newfixationfsjoint = 'newfixationimg/fsjoint10ss'
        self.newsaccadefixationfsjoint = 'newsaccadefixationimg/fsjoint10ss'
        self.newgazepointfsjoint = 'newgazepointimg/fsjoint10ss'
        self.saliency='../Saliency'
        self.newsaliency='../TASED-Net-master/output'
        self.frame='../img/frame'
        self.gazepointfsjoint1d='gazepointimg/fsjoint1d'
        self.newgazepointfsjoint1d = 'newgazepointimg/fsjoint1d'
        self.fixationfsjoint1d='fixationimg/fsjoint1d'
        self.newfixationfsjoint1d='newfixationimg/fsjoint1d'
        self.saccadefixationfsjoint1d='saccadefixationimg/fsjoint1d'
        self.newsaccadefixationfsjoint1d='newsaccadefixationimg/fsjoint1d'

        self.framecoor='framecoordinate'
        self.visualelement='visualelement'
        self.firstfixationtime='stafeatures/firstfixationtime'
        self.aoifeatures='stafeatures/aoi'
        self.stafeatures='stafeatures/stafeatures'
        self.stapath='stapath'

def getParticipants(paramas,TD,file):
    Participants = {}
    pList=os.listdir(f'{paramas.visualelement}/media{file}')
    for x in pList:
        if TD:
            if x[0]=='n':
                fo = open(f'{paramas.visualelement}/media{file}/{x}', "r")
                myFile = fo.read()
                myRecords = myFile.split('\n')
                myRecords_templist = []

                for y in range(1, len(myRecords) - 1):
                    myRecords_templist.append(myRecords[y].split('\t'))  # 把表格数据存入
                Participants[x.split('.')[0]] = myRecords_templist  # 一个人一个字典
        else:
            if x[0]!='n':
                fo = open(f'{paramas.visualelement}/media{file}/{x}', "r")
                myFile = fo.read()
                myRecords = myFile.split('\n')
                myRecords_templist = []

                for y in range(1, len(myRecords) - 1):
                    myRecords_templist.append(myRecords[y].split('\t'))  # 把表格数据存入
                Participants[x.split('.')[0]] = myRecords_templist  # 一个人一个字典
    return Participants


def createSequences(Participants):
    Sequences = {}
    keys = list(Participants.keys())
    for y in range(0,len(keys)):  # 循环参与者
        sequence = ""

        for z in range(0, len(Participants[keys[y]])):  # 参与者的每一行

            tempAoI = Participants[keys[y]][z][9]
            tempDuration =int(Participants[keys[y]][z][5])

            if len(tempAoI) != 0:
                sequence = sequence + tempAoI + "-" + str(tempDuration) + "."
        # print("A sequence has been created for " + keys[y])
        Sequences[keys[y]] = sequence  # 得到所有人的序列，AOI和持续时间
        # print(Sequences[keys[y]])
    return Sequences

def getNumberedSequence(Sequence):
    numberedSequence = []
    numberedSequence.append([Sequence[0][0], 1, Sequence[0][1]])

    for y in range(1, len(Sequence)):  # 连续一样的AOI，他们的号一样
        if Sequence[y][0] == Sequence[y - 1][0]:
            numberedSequence.append([Sequence[y][0], numberedSequence[len(numberedSequence) - 1][1], Sequence[y][1]])
        else:
            numberedSequence.append([Sequence[y][0], getSequenceNumber(Sequence[0:y], Sequence[y][0]), Sequence[y][1]])

    AoIList = getExistingAoIListForSequence(numberedSequence)  # 去掉一样序号的AOI且没有持续时间
    AoINames = ['A','B','C','D','E']
    newSequence = []

    myList = []
    myDictionary = {}
    replacementList = []

    for x in range(0, len(AoIList)):
        totalDuration = 0
        for y in range(0, len(numberedSequence)):
            if numberedSequence[y][0:2] == AoIList[x]:
                totalDuration = totalDuration + int(numberedSequence[y][2])
        myList.append([AoIList[x], totalDuration])  # 去掉一样序号的AOI加上总的持续时间

    for x in range(0, len(AoINames)):
        myAoIList = [w for w in myList if w[0][0] == AoINames[x]]
        myAoIList.sort(key=lambda x: x[1])  # 各个AOI按注视时长升序排列，有时间
        myAoIList.reverse()  # 将列表反向排列，时间最长的在第一个
        if len(myAoIList) > 0:
            myDictionary[AoINames[x]] = myAoIList

    for AoI in AoIList:
        index = [w[0] for w in myDictionary[AoI[0]]].index(AoI)
        replacementList.append([AoI, [AoI[0], (index + 1)]])

    for x in range(0, len(numberedSequence)):
        myReplacementList = [w[0] for w in replacementList]
        index = myReplacementList.index(numberedSequence[x][0:2])
        newSequence.append([replacementList[index][1][0]] + [replacementList[index][1][1]] + [numberedSequence[x][2]])

    return newSequence

def getSequenceNumber(Sequence, Item):  # 该AOI出现的次数
    abstractedSequence = getAbstractedSequence(Sequence)
    return abstractedSequence.count(Item) + 1


def getAbstractedSequence(Sequence):  # 连续不一样的AOI序列
    myAbstractedSequence = [Sequence[0][0]]
    for y in range(1, len(Sequence)):
        if myAbstractedSequence[len(myAbstractedSequence) - 1] != Sequence[y][0]:
            myAbstractedSequence.append(Sequence[y][0])
    return myAbstractedSequence

def getExistingAoIListForSequence(Sequence):  # 去掉一样标号的AOI
    AoIlist = []
    for x in range(0, len(Sequence)):
        try:
            AoIlist.index(Sequence[x][0:2])  # 找不到索引时抛出异常
        except:
            AoIlist.append(Sequence[x][0:2])
    return AoIlist


def calculateImportanceThreshold(mySequences, Threshold):  #最小出现频率、最小持续时间
    myAoICounter = getNumberDurationOfAoIs(mySequences)  #AOI 标号 counter 时间 flagCounter
    commonAoIs = []
    for myAoIdetail in myAoICounter:
        if myAoIdetail[3] >= Threshold:
            commonAoIs.append(myAoIdetail)

    if len(commonAoIs) == 0:
        return -1

    minValueCounter = commonAoIs[0][1]
    for AoIdetails in commonAoIs:
        if minValueCounter > AoIdetails[1]:
            minValueCounter = AoIdetails[1]

    minValueDuration = commonAoIs[0][2]
    for AoIdetails in commonAoIs:
        if minValueDuration > AoIdetails[2]:
            minValueDuration = AoIdetails[2]

    return [minValueCounter, minValueDuration]


def getNumberDurationOfAoIs(Sequences):  #【AOI 标号】 counter 时间 flagCounter
    AoIs = getExistingAoIList(Sequences)  # 得到所有人的连续不一致的AOI序列， AOI 标号
    AoIcount = []
    for x in range(0, len(AoIs)):
        counter = 0
        duration = 0
        flagCounter = 0
        keys = list(Sequences.keys())
        for y in range(0, len(keys)):
            if [s[0:2] for s in Sequences[keys[y]]].count(AoIs[x]) > 0:
                counter = counter + [s[0:2] for s in Sequences[keys[y]]].count(AoIs[x])
                duration = duration + sum([int(w[2]) for w in Sequences[keys[y]] if w[0:2] == AoIs[x]])
                flagCounter = flagCounter + 1   #当前路径包含AOI 标号 就加1

        AoIcount.append([AoIs[x], counter, duration, flagCounter])

    return AoIcount

def updateAoIsFlag(AoIs, threshold):  #是否可作为趋势实例
    for AoI in AoIs:
        if AoI [1] >= threshold[0] and AoI [2] >= threshold[1]:
            AoI [3] = True
    return AoIs

def removeInsignificantAoIs(Sequences, AoIList):  #删除非趋势实例
    sequences = Sequences.copy()
    significantAoIs = []
    for AoI in AoIList:
        if AoI [3] == True:
            significantAoIs.append(AoI[0])

    keys = list(sequences.keys())
    for y in range (0 , len (keys)):
        temp = []
        for k in range (0, len(sequences[keys[y]])):
            try:
                significantAoIs.index(sequences[keys[y]][k][0:2])
                temp.append(sequences[keys[y]][k])
            except:
                continue
        sequences[keys[y]] = temp
    return sequences

def getExistingAoIList(Sequences):  # 得到所有人的连续不一致的AOI序列，有标号，无时间
    AoIlist = []
    keys = list(Sequences.keys())
    for y in range(0, len(keys)):
        for x in range(0, len(Sequences[keys[y]])):
            try:
                AoIlist.index(Sequences[keys[y]][x][0:2])
            except:
                AoIlist.append(Sequences[keys[y]][x][0:2])
    return AoIlist


def calculateNumberDurationOfFixationsAndNSV(Sequences):
    keys = list(Sequences.keys())
    for x in range(0, len(keys)):
        myAbstractedSequence = []
        if len(Sequences[keys[x]]) != 0:
            myAbstractedSequence = [Sequences[keys[x]][0][0:2] + [1] + [int(Sequences[keys[x]][0][2])]]  #AOI 标号 次数 总duration
            for y in range(1, len(Sequences[keys[x]])):
                if myAbstractedSequence[len(myAbstractedSequence) - 1][0:2] != Sequences[keys[x]][y][0:2]:
                    myAbstractedSequence.append(Sequences[keys[x]][y][0:2] + [1] + [int(Sequences[keys[x]][y][2])])
                else:
                    myAbstractedSequence[len(myAbstractedSequence) - 1][2] = myAbstractedSequence[len(myAbstractedSequence) - 1][2] + 1
                    myAbstractedSequence[len(myAbstractedSequence) - 1][3] = myAbstractedSequence[len(myAbstractedSequence) - 1][3] + int(Sequences[keys[x]][y][2])

        Sequences[keys[x]] = myAbstractedSequence

    keys = list(Sequences.keys())
    for x in range(0, len(keys)):
        for y in range(0, len(Sequences[keys[x]])):
            if len(Sequences[keys[x]]) < 2:
                value = 0
            else:
                value = 0.9 / (len(Sequences[keys[x]]) - 1)
            NSV = 1 - round(y, 2) * value
            Sequences[keys[x]][y] = Sequences[keys[x]][y] + [NSV]  #AOI 标号 次数 总duration NSV
    return Sequences

def calculateTotalNumberDurationofFixationsandNSV(AoIList, Sequences):
    for x in range (0, len(AoIList)):
        duration = 0
        counter = 0
        totalNSV = 0

        flag = 0
        keys = list(Sequences.keys())
        for y in range (0 , len (keys)):
             for k in range (0, len (Sequences[keys[y]])):
                 if Sequences[keys[y]][k][0:2] == AoIList[x]:
                     counter = counter + Sequences[keys[y]][k][2]
                     duration = duration + Sequences[keys[y]][k][3]
                     totalNSV = totalNSV + Sequences[keys[y]][k][4]
                     flag = flag + 1

        AoIList[x] = AoIList[x] + [counter] + [duration]  + [totalNSV] + [flag]
    return AoIList


def getValueableAoIs(AoIList, Threshold):
    commonAoIs = []
    valuableAoIs = []
    for myAoIdetail in AoIList:
        if myAoIdetail[5] >= Threshold:
            commonAoIs.append(myAoIdetail)

    minValue = commonAoIs[0][4]
    for AoIdetails in commonAoIs:
        if minValue > AoIdetails[4]:
            minValue = AoIdetails[4]

    for myAoIdetail in AoIList:
        if myAoIdetail[4] >= minValue:
            valuableAoIs.append(myAoIdetail)

    return valuableAoIs


# Function getAbstractedSequences definition is here
def getAbstractedSequences(Sequences):  #AOI（不重复） counter（累加）
    n_Sequences = {}
    keys = list(Sequences.keys())
    for x in range(0, len(keys)):
        myAbstractedSequence = []
        if len(Sequences[keys[x]]) != 0:
            myAbstractedSequence = [Sequences[keys[x]][0][0:1] + [int(Sequences[keys[x]][0][1])]]
            for y in range(1, len(Sequences[keys[x]])):
                if myAbstractedSequence[len(myAbstractedSequence) - 1][0:1] != Sequences[keys[x]][y][0:1]:
                    myAbstractedSequence.append(Sequences[keys[x]][y][0:1] + [int(Sequences[keys[x]][y][1])])
                else:
                    myAbstractedSequence[len(myAbstractedSequence) - 1][1] = \
                    myAbstractedSequence[len(myAbstractedSequence) - 1][1] + int(Sequences[keys[x]][y][1])

        n_Sequences[keys[x]] = myAbstractedSequence
    return n_Sequences


def getStringEditDistance(Sequence1, Sequence2):
    distance = 0
    matrix = []

    for k in range(0, len(Sequence1) + 1):
        matrix.append([])
        for g in range(0, len(Sequence2) + 1):
            matrix[k].append(0)

    for k in range(0, len(Sequence1) + 1):
        matrix[k][0] = k

    for g in range(0, len(Sequence2) + 1):
        matrix[0][g] = g

    for g in range(1, len(Sequence2) + 1):
        for k in range(1, len(Sequence1) + 1):
            if Sequence1[k - 1] == Sequence2[g - 1]:
                matrix[k][g] = min(matrix[k - 1][g - 1] + 0, matrix[k][g - 1] + 1, matrix[k - 1][g] + 1)
            else:
                matrix[k][g] = min(matrix[k - 1][g - 1] + 1, matrix[k][g - 1] + 1, matrix[k - 1][g] + 1)
    distance = matrix[len(Sequence1)][len(Sequence2)]
    return distance


# Function calculateAverageSimilarity definition here
def calculateAverageSimilarity(Sequences, commonSequence):
    distancelist = []
    keys = list(Sequences.keys())
    for y in range(0, len(keys)):
        distance = getStringEditDistance(Sequences[keys[y]], commonSequence)
        normalisedScore = distance / float(max(len(Sequences[keys[y]]), len(commonSequence)))
        similarity = 100.0 * (1 - normalisedScore)
        distancelist.append(similarity)
    return np.median(distancelist)

# STA Algorithm

# Preliminary Stage
pa=paramas()
trendTD = []
trendASD=[]
for c in range(1,15):
    toleranceLevel = [] #Provide the tolerance level between 0 and 1 [0.00, 0.01...1.00]
    for i in range(101):
        toleranceLevel.append(round(i/100,2))
    highestFidelity =  True#Find an find an appropriate tolerance level for achieving the highest fidelity to individual scanpaths based on the input.

    myParticipants = getParticipants(pa,True,c)
    mySequences = createSequences(myParticipants)

    keys = list(mySequences.keys())
    for y in range(0, len(keys)):
        mySequences[keys[y]] = mySequences[keys[y]].split('.')
        del mySequences[keys[y]][len(mySequences[keys[y]]) - 1]
    for y in range(0, len(keys)):
        for z in range(0, len(mySequences[keys[y]])):
            mySequences[keys[y]][z] = mySequences[keys[y]][z].split('-')
    a=0
    b=0
    for y in range(0,len(keys)):
        if len(mySequences[keys[y]])==0:
            print(keys[y])
            a=a+1
            del mySequences[keys[y]]
        # elif len(mySequences[keys[y]])==1:
        #     print('only 1 element in ' + keys[y])
        #     b=b+1
    print('0 element:'+str(a))
    #         del mySequences[keys[y]]

    # First-Pass
    mySequences_num = {}
    keys = list(mySequences.keys())
    for y in range(0, len(keys)):
        if (len(mySequences[keys[y]]) != 0):
            mySequences_num[keys[y]] = getNumberedSequence(mySequences[keys[y]])
        else:
            mySequences_num[keys[y]] = []

    if highestFidelity is not True:
        ToleranceThreshold = toleranceLevel * len(keys)
        myImportanceThreshold = calculateImportanceThreshold(mySequences_num, ToleranceThreshold)
        if myImportanceThreshold != -1:
            myImportantAoIs = updateAoIsFlag(getNumberDurationOfAoIs(mySequences_num), myImportanceThreshold)
            myNewSequences = removeInsignificantAoIs(mySequences_num, myImportantAoIs)
            if myNewSequences == -1:
                print("Trending Path:", [])
            else:
                # Second-Pass
                myNewAoIList = getExistingAoIList(myNewSequences)
                myNewAoIList = calculateTotalNumberDurationofFixationsandNSV(myNewAoIList,
                                                                             calculateNumberDurationOfFixationsAndNSV(
                                                                                 myNewSequences))
                myFinalList = getValueableAoIs(myNewAoIList, ToleranceThreshold)
                myFinalList.sort(key=lambda x: (x[4], x[3], x[2]))
                myFinalList.reverse()

                commonSequence = []
                for y in range(0, len(myFinalList)):
                    commonSequence.append(myFinalList[y][0])

                trendingPath = getAbstractedSequence(commonSequence)
                print("Trending Path:", trendingPath)
        else:
            print("Trending Path:", [])
    else:
        tolerantPaths = []
        for toleranceLevel in [float(j) / 100 for j in range(0, 101)]:
            ToleranceThreshold = toleranceLevel * len(keys)
            myImportanceThreshold = calculateImportanceThreshold(mySequences_num, ToleranceThreshold)
            if myImportanceThreshold != -1:
                myImportantAoIs = updateAoIsFlag(getNumberDurationOfAoIs(mySequences_num), myImportanceThreshold)
                myNewSequences = removeInsignificantAoIs(mySequences_num, myImportantAoIs)
                # Second-Pass
                myNewAoIList = getExistingAoIList(myNewSequences)
                myNewAoIList = calculateTotalNumberDurationofFixationsandNSV(myNewAoIList,
                                                                             calculateNumberDurationOfFixationsAndNSV(
                                                                                 myNewSequences))
                myFinalList = getValueableAoIs(myNewAoIList, ToleranceThreshold)
                myFinalList.sort(key=lambda x: (x[4], x[3], x[2]))
                myFinalList.reverse()

                commonSequence = []
                for y in range(0, len(myFinalList)):
                    commonSequence.append(myFinalList[y][0])

                trendingPath = getAbstractedSequence(commonSequence)

                myNewNormalSequences_Temp = {}
                myNewNormalSequences_Temp = getAbstractedSequences(mySequences)

                keys = list(myNewNormalSequences_Temp.keys())
                for y in range(0, len(keys)):
                    tempSequence = []
                    for z in range(0, len(myNewNormalSequences_Temp[keys[y]])):
                        tempSequence.append(myNewNormalSequences_Temp[keys[y]][z][0])
                    myNewNormalSequences_Temp[keys[y]] = getAbstractedSequence(tempSequence)

                tolerantPaths.append(
                    [trendingPath, calculateAverageSimilarity(myNewNormalSequences_Temp, trendingPath), toleranceLevel])
            else:
                tolerantPaths.append([[], calculateAverageSimilarity(myNewNormalSequences_Temp, []), toleranceLevel])
        tolerantPaths.sort(key=lambda x: x[1])
        tolerantPaths.reverse()
        print("Trending Path:", tolerantPaths[0][0])
        print("Tolerance Level:", tolerantPaths[0][2])
        trendTD.append(['media'+str(c),tolerantPaths[0][0]])
for d in range(1, 15):
    toleranceLevel = []  # Provide the tolerance level between 0 and 1 [0.00, 0.01...1.00]
    for i in range(101):
        toleranceLevel.append(round(i / 100, 2))
    highestFidelity = True  # Find an find an appropriate tolerance level for achieving the highest fidelity to individual scanpaths based on the input.

    myParticipants = getParticipants(pa,False,d)
    mySequences = createSequences(myParticipants)

    keys = list(mySequences.keys())
    for y in range(0, len(keys)):
        mySequences[keys[y]] = mySequences[keys[y]].split('.')
        del mySequences[keys[y]][len(mySequences[keys[y]]) - 1]
    for y in range(0, len(keys)):
        for z in range(0, len(mySequences[keys[y]])):
            mySequences[keys[y]][z] = mySequences[keys[y]][z].split('-')
    a = 0
    b = 0
    for y in range(0, len(keys)):
        if len(mySequences[keys[y]]) == 0:
            print(keys[y])
            a = a + 1
            del mySequences[keys[y]]
        # elif len(mySequences[keys[y]])==1:
        #     print('only 1 element in ' + keys[y])
        #     b=b+1
    print('0 element:' + str(a))
    #         del mySequences[keys[y]]

    # First-Pass
    mySequences_num = {}
    keys = list(mySequences.keys())
    for y in range(0, len(keys)):
        if (len(mySequences[keys[y]]) != 0):
            mySequences_num[keys[y]] = getNumberedSequence(mySequences[keys[y]])
        else:
            mySequences_num[keys[y]] = []

    if highestFidelity is not True:
        ToleranceThreshold = toleranceLevel * len(keys)
        myImportanceThreshold = calculateImportanceThreshold(mySequences_num, ToleranceThreshold)
        if myImportanceThreshold != -1:
            myImportantAoIs = updateAoIsFlag(getNumberDurationOfAoIs(mySequences_num), myImportanceThreshold)
            myNewSequences = removeInsignificantAoIs(mySequences_num, myImportantAoIs)
            if myNewSequences == -1:
                print("Trending Path:", [])
            else:
                # Second-Pass
                myNewAoIList = getExistingAoIList(myNewSequences)
                myNewAoIList = calculateTotalNumberDurationofFixationsandNSV(myNewAoIList,
                                                                             calculateNumberDurationOfFixationsAndNSV(
                                                                                 myNewSequences))
                myFinalList = getValueableAoIs(myNewAoIList, ToleranceThreshold)
                myFinalList.sort(key=lambda x: (x[4], x[3], x[2]))
                myFinalList.reverse()

                commonSequence = []
                for y in range(0, len(myFinalList)):
                    commonSequence.append(myFinalList[y][0])

                trendingPath = getAbstractedSequence(commonSequence)
                print("Trending Path:", trendingPath)
        else:
            print("Trending Path:", [])
    else:
        tolerantPaths = []
        for toleranceLevel in [float(j) / 100 for j in range(0, 101)]:
            ToleranceThreshold = toleranceLevel * len(keys)
            myImportanceThreshold = calculateImportanceThreshold(mySequences_num, ToleranceThreshold)
            if myImportanceThreshold != -1:
                myImportantAoIs = updateAoIsFlag(getNumberDurationOfAoIs(mySequences_num), myImportanceThreshold)
                myNewSequences = removeInsignificantAoIs(mySequences_num, myImportantAoIs)
                # Second-Pass
                myNewAoIList = getExistingAoIList(myNewSequences)
                myNewAoIList = calculateTotalNumberDurationofFixationsandNSV(myNewAoIList,
                                                                             calculateNumberDurationOfFixationsAndNSV(
                                                                                 myNewSequences))
                myFinalList = getValueableAoIs(myNewAoIList, ToleranceThreshold)
                myFinalList.sort(key=lambda x: (x[4], x[3], x[2]))
                myFinalList.reverse()

                commonSequence = []
                for y in range(0, len(myFinalList)):
                    commonSequence.append(myFinalList[y][0])

                trendingPath = getAbstractedSequence(commonSequence)

                myNewNormalSequences_Temp = {}
                myNewNormalSequences_Temp = getAbstractedSequences(mySequences)

                keys = list(myNewNormalSequences_Temp.keys())
                for y in range(0, len(keys)):
                    tempSequence = []
                    for z in range(0, len(myNewNormalSequences_Temp[keys[y]])):
                        tempSequence.append(myNewNormalSequences_Temp[keys[y]][z][0])
                    myNewNormalSequences_Temp[keys[y]] = getAbstractedSequence(tempSequence)

                tolerantPaths.append(
                    [trendingPath, calculateAverageSimilarity(myNewNormalSequences_Temp, trendingPath), toleranceLevel])
            else:
                tolerantPaths.append([[], calculateAverageSimilarity(myNewNormalSequences_Temp, []), toleranceLevel])
        tolerantPaths.sort(key=lambda x: x[1])
        tolerantPaths.reverse()
        print("Trending Path:", tolerantPaths[0][0])
        print("Tolerance Level:", tolerantPaths[0][2])
        trendASD.append(tolerantPaths[0][0])
f = xlwt.Workbook()
sheet3 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
# 将数据写入第 i 行，第 j 列
row0 = [ 'Media_name', 'TD_Path','length','ASD_Path','length','StringEditDistance','similarity']
for w in range(0, len(row0)):
    sheet3.write(0, w, row0[w])
distance=[]

for i in range(len(trendTD)):
    D=getStringEditDistance(trendTD[i][1],trendASD[i])
    if len(trendTD[i][1])>len(trendASD[i]):
        L=len(trendTD[i][1])
    else:
        L=len(trendASD[i])
    s=1-D/L
    distance.append([D,s])



for da in range(len(trendTD)):
    sheet3.write(da+1,0,trendTD[da][0])
    sheet3.write(da + 1, 1, trendTD[da][1])
    sheet3.write(da+1,2,len(trendTD[da][1]))
    sheet3.write(da + 1, 3, trendASD[da])
    sheet3.write(da + 1, 4, len(trendASD[da]))
    sheet3.write(da + 1, 5, distance[da][0])
    sheet3.write(da + 1, 6, distance[da][1])


if not os.path.exists(pa.stapath):
    os.makedirs(pa.stapath)
f.save(pa.stapath  + '/trendPath2.xls')  # 保存文件

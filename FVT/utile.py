# -*- coding: utf-8 -*-
#general
randomSeed=0
datasetPath='setA'
cuda_device = "0"#控制当前使用第几块GPU，从0开始计数


#net

#train
batchsizeForTrain=256#batchsizeForTrain*12才是实际的batch数量
epochMax=2000
gamma = 0.01
weight_cent = 0.1
modelName='FVT_' + datasetPath

if __name__=='__main__':
    print(modelName)
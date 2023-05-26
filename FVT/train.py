import torch
from torchvision import transforms
import numpy as np
import time
from net import Net

from cos_center_loss import CosCenterLoss
from utile import cuda_device as cuda_device
from utile import modelName,gamma,weight_cent
from utile import datasetPath,randomSeed
from utile import batchsizeForTrain,epochMax
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path=os.path.join('..','..','data',datasetPath)
print('-'*25,'loading','-'*25)
#初始化种子
def init_seeds(seed=0):
    torch.manual_seed(seed)
    # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers for the current GPU.
    # It’s safe to call this function if CUDA is not available;
    # in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)
    # Sets the seed for generating random numbers on all GPUs.
    # It’s safe to call this function if CUDA is not available;
    # in that case, it is silently ignored.
    np.random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
init_seeds(randomSeed)

#读取数据集
data = np.load(os.path.join(path,'data.npy'))
print('origin dataset :', data.shape)
userCount = data.shape[0]
label = torch.flip(torch.arange(userCount), dims = [0]).unsqueeze(1).expand(userCount, data.shape[1])
data = data.reshape(-1,1,data.shape[-2],data.shape[-1])
label = label.reshape(-1)

testData=data[0:data.shape[0]//2,:,:,:]#first part of data
testLabel=label[0:data.shape[0]//2]
trainData=data[data.shape[0]//2:,:,:,:]#second part of data
trainLabel=label[data.shape[0]//2:]
#转为dataset
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
trainset = torch.utils.data.TensorDataset(transform(torch.from_numpy(trainData).type(torch.FloatTensor)),
                                          trainLabel)
testset = torch.utils.data.TensorDataset(transform(torch.from_numpy(testData).type(torch.FloatTensor)),
                                         testLabel)
#转为loader
loader_training = torch.utils.data.DataLoader(trainset, batch_size=batchsizeForTrain, shuffle=True)
loader_test = torch.utils.data.DataLoader(testset, batch_size=batchsizeForTrain, shuffle=False)

#加载net
model = Net(userCount // 2).to(device)
optimizer = torch.optim.SGD(model.parameters(),
                lr=0.001,
                momentum=0.9,
                weight_decay=0.01,
                nesterov=False)
centerLoss = CosCenterLoss(num_classes=userCount // 2, feat_dim=64, use_gpu=True if device == 'cuda' else False)
catagoricialLoss = torch.nn.CrossEntropyLoss()
optimizerCL = torch.optim.SGD(centerLoss.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=0.01,
                nesterov=False)

def train(epoch):
    print(time.strftime("[%H:%M:%S]", time.localtime()),
          '\tEpoch_train: %d' % epoch, end = ' ')
    startTime = time.time()
    correct, total = 0, 0
    model.train()
    for batch_idx, (data, label) in enumerate(loader_training):
        data = data.type(torch.FloatTensor).to(device)
        label = label.to(device)
        # print(data.shape, label.shape), print(label[0:10])
        out, feature = model(data)
        predictions = out.data.max(1)[1]
        total += label.size(0)
        correct += (predictions == label.data).sum()
        loss1 = catagoricialLoss(out, label)
        loss2 = gamma * centerLoss(feature, label)
        loss = loss1 + gamma * loss2
        optimizer.zero_grad()
        optimizerCL.zero_grad()
        loss.backward()
        optimizer.step()
        # 权重
        for param in centerLoss.parameters():
            param.grad.data *= weight_cent
        optimizerCL.step()
    acc = correct * 100. / total
    print('<correct %d/%d; accuracy: %.2f%%>' % (correct, total, acc.item()), end=' ')
    print('\t|| over %.2f s'%(time.time() - startTime))

def generateFeature():
    print(time.strftime("[%H:%M:%S]", time.localtime()),
          '\tGenerate: feature', end = ' ')
    startTime = time.time()
    model.eval()
    featureGenerated = None
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader_test):
            data = data.type(torch.FloatTensor).to(device)
            _, feature = model(data)
            if featureGenerated == None:
                featureGenerated = feature
            else:
                featureGenerated = torch.cat((featureGenerated, feature), dim = 0)
    # print('featureGenerated.shape: ', featureGenerated.shape)
    print('\t|| over %.2f s' % (time.time() - startTime))
    return featureGenerated
def getGenImp(featureGenerated):
    print(time.strftime("[%H:%M:%S]", time.localtime()),
          '\tGenerate: genuine, imposter', end = ' ')
    startTime = time.time()
    splitNum = 3 # 分组, 组内计算imposter
    per = featureGenerated.shape[0] // splitNum
    genuine, imposter = None, None
    for i in range(splitNum):
        data = featureGenerated[i * per : (i + 1) * per]
        data = torch.nn.functional.cosine_similarity(data.unsqueeze(1), data.unsqueeze(0), dim=-1, eps=1e-08)
        data = (data + 1) / 2
        countPerUser = 10
        for j in range(0, data.shape[0], countPerUser):
            gen = data[j:j + 10, j:j + 10]
            imp = torch.cat((data[j:j + 10, :j], data[j:j + 10, j + 10:]),dim = -1)
            if genuine == None:
                genuine = gen
            else:
                genuine = torch.cat((genuine, gen), dim = 0)
            if imposter == None:
                imposter = imp
            else:
                imposter = torch.cat((imposter, imp), dim = 0)
    print('\t|| over %.2f s'%(time.time() - startTime))
    return genuine, imposter
def getFAR_FRR(genuine, imposter):
    FAR, FRR=[],[]
    for t in range(10,10000,10):
        t = t / 10000.0
        far = ((imposter > t).sum() / (imposter.shape[0] * imposter.shape[1])).item()
        frr = ((genuine < t).sum() / (genuine.shape[0] * (genuine.shape[1]))).item()
        FAR.append(far), FRR.append(frr)
    FAR, FRR = np.array(FAR), np.array(FRR)
    cap = np.abs(FAR - FRR)
    Index = np.argmin(cap)
    eer = (FAR[Index]+FRR[Index])/2
    print('eer:%.4f'%eer)
    return FAR, FRR
def checkEER(epoch):
    print(time.strftime("[%H:%M:%S]", time.localtime()),
          '\tEpoch_CheckEER: %d' % epoch)
    startTime = time.time()
    feature = generateFeature()
    genuine, imposter = getGenImp(feature)
    FAR, FRR = getFAR_FRR(genuine, imposter)
    if not os.path.exists('output'):
        os.makedirs('output')
    name = os.path.join('output',modelName)
    torch.save(feature, name + '_feature.pt')
    torch.save(genuine, name + '_genuine.pt')
    torch.save(imposter, name + '_imposter.pt')
    np.save(name + '_FAR.npy', FAR)
    np.save(name + '_FRR.npy', FRR)
    print(time.strftime("[%H:%M:%S]", time.localtime()),
          '\tEpoch_CheckEER: %d' % epoch, end = ' ')
    print('\t|| over %.2f s' % (time.time() - startTime))
def run():
    # epochMax = 1
    if not os.path.exists('model'):
        os.makedirs('model')
    name = os.path.join('model',modelName + '.pkl')
    for epoch in range(1, 1 + epochMax):
        print('=' * 40, epoch, '=' * 40)
        train(epoch)
        checkEER(epoch)
    torch.save(model.state_dict(), name)
if __name__=='__main__':
    run()



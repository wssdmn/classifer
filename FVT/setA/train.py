import sys
sys.path.append("..")
from net import Net as Net
from cos_center_loss import CosCenterLoss
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import shutil
import time

datasetPath = 'setA'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('-' * 25, 'loading', '-' * 25)

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

batchsizeForTrain, batchsizeForValid, batchsizeForTest = 32, 50, 50
START_FROM_SCRATCH = False
MAX_EPOCH = 1000
SAVE_EPOCH_LIST = [50, 100, 200, 400, 500, 600, 700, 800, MAX_EPOCH]

init_seeds(0)
path = os.path.join('..', '..', 'data', datasetPath)
data = np.load(os.path.join(path, 'data.npy'))
data = torch.from_numpy(data)
user_count = data.shape[0]
label = torch.arange(user_count).unsqueeze(1).expand(user_count, data.shape[1])
data = data.reshape(-1,1,data.shape[-2],data.shape[-1])
label = label.reshape(-1)
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,)),
                                transforms.Resize([128, 128])])
dataset = torch.utils.data.TensorDataset(transform(data.type(torch.FloatTensor)), label)
train_size = int(len(dataset) * 0.6 * 0.7)
valid_size = int(len(dataset) * 0.6 * 0.3)
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
print(len(train_dataset), len(valid_dataset), len(test_dataset))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getDataTrain():
    # 实现用于训练的数据的加载。训练集是用来训练shared_cnn的。
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batchsizeForTrain, shuffle=True)
    return dataloader_train


def getDataValid():
    # 实现用于验证的数据的加载。验证集是用来训练controller的。
    dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsizeForValid, shuffle=True)
    return dataloader_valid


def getDataTest():
    # 实现用于测试的数据的加载。测试集是两个网络都训练好了过后，用来评估性能的。
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batchsizeForTest, shuffle=False)
    return dataloader_test


def loadNet(start_from_scratch=True, name='Arcvein'):
    net = Net(user_count).to(device)
    model_folder_path = name + '_model'
    model_name = os.path.join(model_folder_path, 'net.pkl')
    if start_from_scratch == False:
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        if os.path.exists(model_name):
            net.load_state_dict(torch.load(model_name))
    centerLossNet = CosCenterLoss(num_classes=user_count, feat_dim=64, use_gpu=True if device == 'cuda' else False)
    model_name = os.path.join(model_folder_path, 'net_cl.pkl')
    if start_from_scratch == False:
        if os.path.exists(model_name):
            centerLossNet.load_state_dict(torch.load(model_name))
    return net,centerLossNet


def loadEpoch(name='Arcvein'):
    start_epoch = 0
    model_folder_path = name + '_model'
    var_name = os.path.join(model_folder_path, 'epoch.npy')
    if os.path.exists(model_folder_path) and os.path.exists(var_name):
        start_epoch = np.load(var_name)
    return start_epoch


def loadBestValidAcc(name='Arcvein'):
    BestValidAcc = 0
    model_folder_path = name + '_model'
    var_name = os.path.join(model_folder_path, 'best_valid_acc.npy')
    if os.path.exists(model_folder_path) and os.path.exists(var_name):
        BestValidAcc = np.load(var_name)
    return BestValidAcc


def saveBestValidAcc(BestValidAcc=None,name='Arcvein'):
    assert BestValidAcc is not None
    model_folder_path = name + '_model'
    np.save(os.path.join(model_folder_path, 'best_valid_acc.npy'), BestValidAcc)


def load_optimizer(net, centerLossNet, start_from_scratch=True, name='Arcvein'):
    model_folder_path = name + '_model'
    opt_file_name = os.path.join(model_folder_path, 'optimizer.pkl')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3,
                                 betas=(0.9, 0.99), eps=1e-08,
                                 weight_decay=5e-4, amsgrad=False)
    optimizerCL = torch.optim.Adam(centerLossNet.parameters(), lr=0.5,
                                   betas=(0.9, 0.99), eps=1e-08,
                                   weight_decay=5e-4, amsgrad=False)
    # optimizer = torch.optim.SGD(net.parameters(),
    #             lr=0.001,
    #             momentum=0.9,
    #             weight_decay=0.01,
    #             nesterov=False)
    # optimizerCL = torch.optim.SGD(centerLossNet.parameters(),
    #                 lr=0.1,
    #                 momentum=0.9,
    #                 weight_decay=0.01,
    #                 nesterov=False)
    if start_from_scratch == False:
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        if os.path.exists(opt_file_name):
            opt = torch.load(opt_file_name)
            optimizer.load_state_dict(opt['optimizer'])
            optimizerCL.load_state_dict(opt['optimizer_cl'])

    return optimizer,optimizerCL


def save(net=None, centerLossNet=None, optimizer=None, optimizerCL=None, epoch=None, name='Arcvein'):
    assert net is not None and centerLossNet is not None
    assert optimizer is not None and optimizerCL is not None
    assert epoch is not None
    model_folder_path = name + '_model'
    torch.save(net.state_dict(), os.path.join(model_folder_path, 'net.pkl'))
    torch.save(centerLossNet.state_dict(), os.path.join(model_folder_path, 'net_cl.pkl'))
    torch.save({'optimizer': optimizer.state_dict(), 'optimizer_cl':optimizerCL.state_dict()},
               os.path.join(model_folder_path, 'optimizer.pkl'))
    np.save(os.path.join(model_folder_path, 'epoch.npy'), epoch)


def train(epoch, net, centerLossNet, optimizer, optimizerCL, dataloader_train):
    init_seeds(epoch)
    print('[Epoch ' + str(epoch) + ': train Net]')
    loss_func = nn.CrossEntropyLoss()
    train_acc_meter, loss_meter = AverageMeter(), AverageMeter()
    net.train()
    for i, (images, labels) in enumerate(dataloader_train):
        images, labels = images.to(device), labels.to(device)
        net.zero_grad(), centerLossNet.zero_grad()
        pred, feature = net(images)
        gamma, weight_cent = 0.001, 0.1
        loss = loss_func(pred, labels) + gamma * centerLossNet(feature, labels)
        loss.backward()
        optimizer.step()
        # 权重
        for param in centerLossNet.parameters():
            param.grad.data *= weight_cent
        optimizerCL.step()
        train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        train_acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())
    display = '\tacc=%.4f' % (train_acc_meter.avg) + '\tloss=%.6f' % (loss_meter.avg)
    print(display)


def get_eval_accuracy(loader, net):
    """Evaluate a given architecture.

    Args:
        loader: A single data loader.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        sample_arc: The architecture to use for the evaluation.

    Returns:
        acc: Average accuracy.
    """
    total = 0.
    acc_sum = 0.
    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred,_ = net(images)
        acc_sum += torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
        total += pred.shape[0]

    acc = acc_sum / total
    return acc.item()


def eval_model(epoch, net, dataloader_valid, dataloader_test):
    init_seeds(epoch)
    print('[Epoch ' + str(epoch) + ': eval net]')
    net.eval()
    valid_acc = get_eval_accuracy(dataloader_valid, net)
    test_acc = get_eval_accuracy(dataloader_test, net)
    print('valid_accuracy: %.4f' % (valid_acc))
    print('test_accuracy: %.4f' % (test_acc))
    return valid_acc


def main(name='Arcvein'):
    init_seeds(0)
    start_from_scratch = START_FROM_SCRATCH
    start_epoch, max_epoch = loadEpoch(name), MAX_EPOCH
    BestValidAcc = loadBestValidAcc(name)
    net,centerLossNet = loadNet(start_from_scratch, name)
    optimizer,optimizerCL = load_optimizer(net, centerLossNet, start_from_scratch, name)
    dataloader_train, dataloader_valid, dataloader_test = getDataTrain(), getDataValid(), getDataTest()
    model_folder_path = name + '_model'
    for epoch in range(start_epoch, max_epoch):
        print(time.strftime("[%H:%M:%S]", time.localtime()))
        init_seeds(epoch)  # 固定随机种子, 保证可重复性
        train(epoch, net, centerLossNet, optimizer, optimizerCL, dataloader_train)
        valid_acc = eval_model(epoch,  net, dataloader_valid, dataloader_test)
        if valid_acc > BestValidAcc:
            BestValidAcc = valid_acc
            torch.save(net.state_dict(), os.path.join(model_folder_path, 'net_best.pkl'))
            torch.save(centerLossNet.state_dict(), os.path.join(model_folder_path, 'net_cl_best.pkl'))
            np.save(os.path.join(model_folder_path, 'best_valid_acc.npy'), BestValidAcc)
        if epoch + 1 in SAVE_EPOCH_LIST:
            shutil.copy(os.path.join(model_folder_path, 'net_best.pkl'),
                        os.path.join(model_folder_path, 'net_best_%d.pkl' % (epoch + 1)))
            shutil.copy(os.path.join(model_folder_path, 'net_cl_best.pkl'),
                        os.path.join(model_folder_path, 'net_cl_best_%d.pkl' % (epoch + 1)))
        print('[best_valid_fixed]: %.4f' % BestValidAcc)
        save(net=net, centerLossNet=centerLossNet, optimizer=optimizer, optimizerCL=optimizerCL, epoch=epoch+1, name=name)
        saveBestValidAcc(BestValidAcc=BestValidAcc,name=name)
if __name__ == "__main__":
    main(name='FVT') # 通过name区分模型
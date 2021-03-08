import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from deep.dataset import Datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import global_settings as settings
from torch.optim.lr_scheduler import _LRScheduler
from deep.mobilenet import MobileNetv2
from deep.model import Net
from deep.ghost_net import ghost_net
from deep.ShuffleNetV2 import shufflenet_v2_x0_5

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]



def train(epoch):       # 开启训练
    net.train()

    loss_sum = 0.0
    correct = 0.0

    for batch_index,(images,labels) in enumerate(train_set):
        if epoch <= args.warm:           # 第一回合使用warmup学习率策略
            warmup_scheduler.step()
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)        # 前向

        outputs_pred = torch.softmax(outputs,1)
        _, preds = outputs_pred.max(1)
        correct += preds.eq(labels).sum()

        loss = loss_function(outputs,labels)  # 损失
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        n_iter = (epoch - 1) * len(train_set) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_set.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    return loss_sum/(len(train_set.dataset)/args.b),correct.float() / len(train_set.dataset)

def eval_training(epoch):         # 测试集验证
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_set:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net(images)
        loss = loss_function(outputs, labels)
        outputs = torch.softmax(outputs,1)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Adduracy: {:.4f}'.format(
        test_loss / (len(test_set.dataset)/32),
        correct.float() / len(test_set.dataset)
    ))

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(test_set.dataset), epoch)
    writer.add_scalar('Test/Adduracy', correct.float() / len(test_set.dataset), epoch)

    return test_loss / (len(test_set.dataset)/32),correct.float() / len(test_set.dataset)


def load_model(model_path,net,gpu_id=None):          # 加载预训练权重
    if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")
    if model_path is not None:

        pretrained_params = torch.load(model_path,map_location=device)

        pretrained_params= \
            {k: v for k, v in pretrained_params.items() if
             k in net.state_dict().keys() and net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(pretrained_params, strict=False)

        # for k, v in pretrained_params.items():
        #     print(k)
        #     print(v)
        # net.load_state_dict(pretrained_params, strict=False)

        # net.load_state_dict(torch.load(model_path, map_location=device))

    print('Angle device:', device)
    # if net is not None:
    #     # 如果网络计算图和参数是分开保存的，就执行参数加载
    #     net = net.to(device)
    #
    #     try:
    #         sk = {}
    #         for k in net:
    #             sk[k[7:]] = net[k]
    #
    #         net.load_state_dict(sk)
    #     except:
    #         net.load_state_dict(net)

        # net = net
        # print('load model')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="shufflenet", help='net type')
    # parser.add_argument('-net', type=str, default="ghost_net_320_8_new", help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=3, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net = ghost_net(num_classes=751).to(device)
    net = shufflenet_v2_x0_5(num_classes=751).to(device)
    # net = MobileNetv2(num_classes=751).to(device)
    # net = Net(num_classes=751).to(device)

    # model_path = r"deep/checkpoint/mobilenet_v2-b0353104.pth"

    # load_model(model_path,net,0)

    train_path = r"F:\BaiduNetdiskDownload\Market-1501-v15.09.15\pytorch\train_all"            # 数据集存放信息
    test_path = r"F:\BaiduNetdiskDownload\Market-1501-v15.09.15\pytorch\val"

    train_set = Datasets(train_path,True)    # 加载数据集
    test_set = Datasets(test_path)

    train_set = DataLoader(train_set, shuffle=True, num_workers=min([os.cpu_count(), 32, 4]), batch_size=32)
    test_set = DataLoader(test_set, shuffle=True, num_workers=min([os.cpu_count(), 32, 4]), batch_size=32)

    loss_function = nn.CrossEntropyLoss()   # 交叉熵做损失

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.1)  # learning rate decay

    iter_per_epoch = len(train_set)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, 'person')
    results_file = os.path.join(checkpoint_path ,'results.txt')

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    log_dir = os.path.join(
        settings.LOG_DIR, args.net, 'person')
    print(log_dir)
    writer = SummaryWriter(log_dir)
    print("done")
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    # 创建模型保存的文件夹
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    best_path = os.path.join(checkpoint_path, '{type}.pth')
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    file = open(results_file,"w")
    file.write("\t\tepoch\t\t\ttrain_loss\t\ttest_loss\t\ttrain_acc\t\ttest_acc\t\tval_acc\t\tbest_acc")
    file.write("\n")
    file.close()

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train_loss,train_acc = train(epoch)     # 返回训练集平均损失
        test_loss,acc = eval_training(epoch)    # 返回测试集损失和精度
        print("test_set in epoch:{} acc is :{}".format(epoch,acc))
        print()
        val_acc = 0        # 返回验证集损失

        # start to save best performance model after learning rate decay to 0.01
        if best_acc <= acc:        # 按照测试集精度保存

            # checkpoint = {
            #     'net_dict': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(checkpoint, './checkpoint/ckpt.t8')

            torch.save(net.state_dict(), best_path.format(type='best'))
            best_acc = acc

        if not epoch % settings.SAVE_EPOCH:    # 每十个轮次保存一次
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        f = open(results_file,"a")
        f.write("\t\t{}/{}\t\t\t{:.4f}\t\t\t{:.4f}\t\t\t{:.4f}\t\t\t{:.4f}\t\t\t{:.4f}\t\t{:.4f}".format(epoch,settings.EPOCH,train_loss,test_loss,train_acc,acc,val_acc,best_acc))
        # f.write("\t\t"+str(epoch)+"\t\t"+str(train_loss)+"\t\t"+str(test_loss)+"\t\t"+str(train_acc)+"\t\t"+str(acc)+"\t\t"+str(val_acc)+"\t\t"+str(best_acc))
        f.write("\n")
        f.close()
    writer.close()
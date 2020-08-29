import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataloader import load_data, Set_Wrapper
import torch
import torch.nn as nn
from models.VGG16 import VGGnet
from tqdm import tqdm
from time import time
from utils import save_model
import numpy as np
import random

def train(ce_loss, optim, device, net, loader):
    # method will be ignored
    net.train()
    label_list = []
    idx_list = []
    for i, (img, label) in tqdm(enumerate(loader)):
        img_t = img.to(device, dtype=torch.float)
        label_t = label.to(device, dtype=torch.long)

        pred, feature = net(img_t)
        _, idx_t = pred.max(dim=1)

        loss = ce_loss(pred, label_t)
        optim.zero_grad()
        loss.backward()
        optim.step()

        idx = idx_t.cpu().detach().numpy()
        idx_list.append(idx)
        label_list.append(label)

    idxs = np.concatenate(idx_list)
    labels = np.concatenate(label_list)


    corret_inds = idxs==labels
    acc = np.mean(corret_inds)
    acc_0 = np.mean(corret_inds[labels==0])
    acc_1 = np.mean(corret_inds[labels==1])
    correct_num = np.sum(corret_inds)
    print('training:', '0:', acc_0, '1:', acc_1, 'correct_num:', correct_num)

    return loss.item()


def test(device, net, loader):
    net.eval()
    label_list = []
    idx_list = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            img_t = img.to(device, dtype=torch.float)
            label_t = label.to(device, dtype=torch.long)
            pred, _ = net(img_t)
            _, idx_t = pred.max(dim=1)
            idx = idx_t.cpu().detach().numpy()
            idx_list.append(idx)
            label_list.append(label)
    idxs = np.concatenate(idx_list)
    labels = np.concatenate(label_list)

    corret_inds = idxs==labels
    acc = np.mean(corret_inds)
    acc_0 = np.mean(corret_inds[labels==0])
    acc_1 = np.mean(corret_inds[labels==1])
    correct_num = np.sum(corret_inds)
    print('testing', '0:', acc_0, '1:', acc_1, 'correct_num:', correct_num)
    return acc


if __name__ == '__main__':
    batch = 32
    epoches = 5
    parent_folder = 'detectors/saved_models'
    if not os.path.exists(parent_folder):
        os.mkdir(model_path)
    model_path = parent_folder + '/vgg16.pth'




    routes = [i for i in range(76)]
    routes.remove(13)
    random.seed(0)
    random.shuffle(routes)


    train_weather_indexes = [15]
    train_routes = routes[:55]

    test_weather_indexes = [15]
    test_routes = routes[55:]

    train_data_dir = 'collected_data'
    test_data_dir = 'collected_data'

    # get datasets
    x_center_train, y_train = load_data(train_data_dir, train_weather_indexes, train_routes)

    inds_0 = y_train == 0
    inds_1 = y_train == 1
    x_center_train_0 = x_center_train[inds_0]
    y_train_0 = y_train[inds_0]
    x_center_train_1 = x_center_train[inds_1]
    y_train_1 = y_train[inds_1]

    x_center_train_0 = x_center_train_0[::1000]
    y_train_0 = y_train_0[::1000]

    x_center_train = np.concatenate([x_center_train_0, x_center_train_1])
    y_train = np.concatenate([y_train_0, y_train_1])

    inds = np.random.permutation(y_train.shape[0])

    x_center_train = x_center_train[inds]
    y_train = y_train[inds]




    x_center_test, y_test = load_data(test_data_dir, test_weather_indexes, test_routes)

    train_dataset = Set_Wrapper((x_center_train, y_train), train_data_dir)
    test_dataset = Set_Wrapper((x_center_test, y_test), test_data_dir)

    print(y_train.shape, np.sum(y_train==0), np.sum(y_train==1))

    # sample_weights = np.ones_like(y_train)
    # sample_weights[y_train==1] *= 60
    # print(np.mean(sample_weights))
    # samp = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
    # train_loader = DataLoader(train_dataset, batch_size=batch, sampler=samp)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # training
    device = torch.device("cuda")
    net = VGGnet().to(device)
    optim = torch.optim.Adadelta(net.parameters(), lr=0.1)


    best_test_acc = 0
    # 76 leads to predicting everything to be 0 VS 77 leads to predicting everything to be 1
    class_weights = [1, 1]
    # class_weights = [1, 1]
    ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).cuda())
    time_start = time()
    for epoch in range(epoches):
        train_loss = train(ce_loss, optim, device, net, train_loader)
        test_acc = test(device, net, test_loader)
        # writer.add_scalar('plain/train_loss', train_loss, epoch)
        # writer.add_scalar('plain/test_acc', test_acc, epoch)
        print('epoch:', epoch, 'test_acc:', test_acc, 'train_loss:', train_loss, 'time elapsed:',  time()-time_start)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print('saved model')
            save_model(net, model_path)

import gc

import torch
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import task_generator as tg
import os
import math
import argparse
import random
from sklearn.decomposition import PCA

from src.efficient_kan import KAN
import CNNEncoder1
import vit

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=128)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=2)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=1000)
parser.add_argument("-t", "--test_episode", type=int, default=100)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-c", "--cpu", type=int, default=0)
args = parser.parse_args()

FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

feature_encoder = CNNEncoder1.rsnet()
feature_encoder.cuda(GPU)

if os.path.exists(
        str("./models/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
    feature_encoder.load_state_dict(torch.load(
        str("./models/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
    print("load feature encoder success")

gearbox_type_9 = []
gearbox_label_9 = []
gearbox_type_10 = []
gearbox_label_10 = []
gearbox_type_11 = []
gearbox_label_11 = []

for i in range(TEST_EPISODE):
    degrees = random.choice([0, 90, 180, 270])
    metatest_character_folders1 = ['../CWT-1000/gearbox/test/health',
                                   '../CWT-1000/gearbox/test/anomaly/anomalyTYPE9']
    # '../CWT-1000/gearbox/test/anomaly/anomalyTYPE13']
    # '../CWT-1000/gearbox/test/anomaly/anomalyTYPE14']

    task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                           SAMPLE_NUM_PER_CLASS, )

    test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                         shuffle=True, rotation=degrees)

    test_dataloader = iter(test_dataloader)
    test_images, test_labels = next(test_dataloader)
    # print(test_labels)
    # print(test_images.shape)  2，3，84，84

    test_features = feature_encoder(Variable(test_images).cuda(GPU))
    # print(test_features.shape)  # 2,128,28,28

    test_features = test_features.cpu().detach().numpy()
    index = 1 if test_labels[0] != 1 else 0
    test_features = test_features[index]
    # print(test_features.shape)
    gearbox_type_9.append(test_features)
    gearbox_label_9.append(9)

for i in range(TEST_EPISODE):
    degrees = random.choice([0, 90, 180, 270])
    metatest_character_folders1 = ['../CWT-1000/gearbox/test/health',
                                   '../CWT-1000/gearbox/test/anomaly/anomalyTYPE10']
    # '../CWT-1000/gearbox/test/anomaly/anomalyTYPE13']
    # '../CWT-1000/gearbox/test/anomaly/anomalyTYPE14']

    task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                           SAMPLE_NUM_PER_CLASS, )

    test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                         shuffle=True, rotation=degrees)

    test_dataloader = iter(test_dataloader)
    test_images, test_labels = next(test_dataloader)
    # print(test_labels)
    # print(test_images.shape)  2，3，84，84

    test_features = feature_encoder(Variable(test_images).cuda(GPU))
    # print(test_features.shape)  # 2,128,28,28

    test_features = test_features.cpu().detach().numpy()
    index = 1 if test_labels[0] != 1 else 0
    test_features = test_features[index]
    # print(test_features.shape)
    gearbox_type_10.append(test_features)
    gearbox_label_10.append(10)

for i in range(TEST_EPISODE):
    degrees = random.choice([0, 90, 180, 270])
    metatest_character_folders1 = ['../CWT-1000/gearbox/test/health',
                                   '../CWT-1000/gearbox/test/anomaly/anomalyTYPE11']
    # '../CWT-1000/gearbox/test/anomaly/anomalyTYPE13']
    # '../CWT-1000/gearbox/test/anomaly/anomalyTYPE14']

    task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                           SAMPLE_NUM_PER_CLASS, )

    test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                         shuffle=True, rotation=degrees)

    test_dataloader = iter(test_dataloader)
    test_images, test_labels = next(test_dataloader)
    # print(test_labels)
    # print(test_images.shape)  2，3，84，84

    test_features = feature_encoder(Variable(test_images).cuda(GPU))
    # print(test_features.shape)  # 2,128,28,28

    test_features = test_features.cpu().detach().numpy()
    index = 1 if test_labels[0] != 1 else 0
    test_features = test_features[index]
    # print(test_features.shape)
    gearbox_type_11.append(test_features)
    gearbox_label_11.append(11)

gearboxlib = gearbox_type_9 + gearbox_type_10 + gearbox_type_11
labellib = gearbox_label_9 + gearbox_label_10 + gearbox_label_11

gearboxlib = np.array(gearboxlib)
gearboxlib = gearboxlib.reshape(300,-1)
print(gearboxlib)
# pca = PCA(n_components=2)
# gearboxlib = pca.fit_transform(gearboxlib)
# 设置聚类数K
K = 3

# 初始化KMeans对象
kmeans = KMeans(n_clusters=K, random_state=0)

# 对数据进行拟合和预测
kmeans.fit(gearboxlib)
labels = kmeans.predict(gearboxlib)
centroids = kmeans.cluster_centers_

# 打印聚类中心和标签
# print("Cluster centers:")
# print(centroids)
# print("Labels:")
# print(labels)
labellib=[labellib[i]-9 for i in range(len(labellib))]
x=0
for t in range(len(labellib)):
    if labellib[t]==labels[t]:
        x=x+1
print(x/300.0)
# 可视化结果

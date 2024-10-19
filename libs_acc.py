# -------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# -------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random
import scipy as sp
import scipy.stats
import RelationNetwork1
import CNNEncoder1
import vit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random

from src.efficient_kan import KAN
import CNNEncoder1
import vit

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=128)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=2)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=4)
parser.add_argument("-e", "--episode", type=int, default=5)
parser.add_argument("-t", "--test_episode", type=int, default=200)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# 填写测试负载

faulttype = 'gearboxtype8'
root_path = '../Test_Data/Single_fault/'
# test_result_sim = './test_result/similar_score/'
# test_result_acc = './test_result/acc_callback/'
test_result_libs = './test_result/libs_acc/libs0/'


# root_path = '../Test_Data/Mixed_faults/MR/'
# test_result_root = './test_result/Mixed_faults_results/MR_fault/'


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    # return m, h
    return m, se


#

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    # metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders(train_folder,test_folder)

    # Step 2: init neural networks
    print("init neural networks")

    gearbox_feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    gearbox_relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    gearbox_relation_network_2 = KAN([8 * 512, 512, 32, 2])
    motor_feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    motor_relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    motor_relation_network_2 = KAN([8 * 512, 512, 32, 2])
    leftaxlebox_feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    leftaxlebox_relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    leftaxlebox_relation_network_2 = KAN([8 * 512, 512, 32, 2])
    gearbox_feature_encoder.cuda(GPU)
    gearbox_relation_network.cuda(GPU)
    gearbox_relation_network_2.cuda(GPU)
    leftaxlebox_feature_encoder.cuda(GPU)
    leftaxlebox_relation_network.cuda(GPU)
    leftaxlebox_relation_network_2.cuda(GPU)
    motor_feature_encoder.cuda(GPU)
    motor_relation_network.cuda(GPU)
    motor_relation_network_2.cuda(GPU)

    # feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    # feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    # relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    if os.path.exists(
            str("./models0/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        gearbox_feature_encoder.load_state_dict(torch.load(
            str("./models0/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load gearbox feature encoder success")
    if os.path.exists(
            str("./models0/gearbox_relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        gearbox_relation_network.load_state_dict(torch.load(
            str("./models0/gearbox_relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load gearbox relation network success")
    if os.path.exists(
            str("./models0/gearbox_relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        gearbox_relation_network_2.load_state_dict(torch.load(
            str("./models0/gearbox_relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load gearbox relation network2 success")
    if os.path.exists(
            str("./models0/leftaxlebox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        leftaxlebox_feature_encoder.load_state_dict(torch.load(
            str("./models0/leftaxlebox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load leftaxlebox feature encoder success")
    if os.path.exists(
            str("./models0/leftaxlebox_relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        leftaxlebox_relation_network.load_state_dict(torch.load(
            str("./models0/leftaxlebox_relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load leftaxlebox relation network success")
    if os.path.exists(
            str("./models0/leftaxlebox_relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        leftaxlebox_relation_network_2.load_state_dict(torch.load(
            str("./models0/leftaxlebox_relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load leftaxlebox relation network2 success")

    if os.path.exists(
            str("./models0/motor_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        motor_feature_encoder.load_state_dict(torch.load(
            str("./models0/motor_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load motor feature encoder success")
    if os.path.exists(
            str("./models0/motor_relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        motor_relation_network.load_state_dict(torch.load(
            str("./models0/motor_relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load motor relation network success")
    if os.path.exists(
            str("./models0/motor_relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        motor_relation_network_2.load_state_dict(torch.load(
            str("./models0/motor_relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load gearbox relation network2 success")

    print("Testing...gearbox type9 ")
    acc_back = []
    for episode in range(10):
        boundary = 0.9
        ture_result = []
        predict_result = []
        scores_result = []
        total_back_rewards = []
        total_rewards = 0
        accuracies = []
        back_accuracies = []
        with torch.no_grad():
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                metatest_character_folders1 = [root_path + 'gearbox/test/health',
                                               root_path + 'gearbox/test/anomaly/anomalyTYPE8']  #
                metatrain_character_folders1 = [root_path + 'gearbox/train/health',
                                                root_path + 'gearbox/train/anomaly0']
                task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                       SAMPLE_NUM_PER_CLASS, )
                task1 = tg.OmniglotTask(metatrain_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                        BATCH_NUM_PER_CLASS, )
                sample_dataloader = tg.get_data_loader(task1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                       shuffle=False, rotation=degrees)
                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                                     shuffle=True, rotation=degrees)
                # sample_images, sample_l0abels = sample_dataloader.__iter__().next()
                # test_images, test_labels = test_dataloader.__iter__().next()
                sample_dataloader = iter(sample_dataloader)
                sample_images, sample_labels = next(sample_dataloader)
                test_dataloader = iter(test_dataloader)
                test_images, test_labels = next(test_dataloader)
                sample_features = gearbox_feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
                test_features = gearbox_feature_encoder(Variable(test_images).cuda(GPU))  # 20x64
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1,
                                                                          1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 4,
                                                                                             28 *
                                                                                             28)
                relations1 = gearbox_relation_network(relation_pairs)
                relations1 = relations1.view(2, 8 * 512)
                relations1 = gearbox_relation_network_2(relations1)
                relations = relations1.view(-1, CLASS_NUM)
                bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)

                for j in range(len(relations)):
                    if relations[j][0] >= boundary:
                        bb[j] = 0
                        scores_result.append(relations[j][0].cpu().item())  # 保存相似分数
                        predict_result.append(0)
                        # print(test_labels[j])# 保存预测结果
                        ture_result.append(test_labels[j].cpu().item())  # 保存真实结果
                    else:
                        bb[j] = 1
                        scores_result.append(relations[j][0].cpu().item())  # 保存相似分数
                        predict_result.append(1)
                        # print(test_labels[j])# 保存预测结果
                        ture_result.append(test_labels[j].cpu().item())  # 保存真实结果
                predict_labels = bb.cpu()
                # print(predict_labels)
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
                # if test_labels[0]==1:
                #     back_reward=[1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
                #     back_acc = np.sum(back_reward) / 1.0 / CLASS_NUM / SAMPLE_NUM_PER_CLASS
                #     back_accuracies.append(back_acc)

                total_rewards += np.sum(rewards)
                accuracy = np.sum(rewards) / 1.0 / CLASS_NUM / SAMPLE_NUM_PER_CLASS
                accuracies.append(accuracy)

            pass
        test_accuracy, std = mean_confidence_interval(accuracies)
        # acc_test_accuracy, std = mean_confidence_interval(back_accuracies)

        acc_back.append(test_accuracy)
        print(acc_back)
        # print("test accuracy:", test_accuracy)
        # np.savetxt(test_result + 'Left' + '_ture_result.csv', np.array(ture_result), fmt='%.4f', delimiter=',')
        # np.savetxt(test_result_sim + faulttype + '_scores.csv', np.array(scores_result), fmt='%.4f', delimiter=',')
        # np.savetxt(test_result + 'Left' + '_predict_result.csv', np.array(predict_result), fmt='%.4f', delimiter=',')
    # tal_accuracy, std = mean_confidence_interval(total_accuracy)

    # print("boundary:", boundary, " aver_accuracy:", aver_accuracy / 10.0, "std:", std)
    # type9_std_acc.append([boundary, aver_accuracy / 10.0, std])
    # np.savetxt(test_result + faulttype + '_accuracy_std.csv', np.array(type9_std_acc), fmt='%.4f', delimiter=',')
    np.savetxt(test_result_libs + faulttype + '_libs0_acc.csv', np.array(acc_back), fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    main()

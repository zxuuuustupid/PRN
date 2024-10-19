
# -------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# -------------------------------------
import gc

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
parser.add_argument("-e", "--episode", type=int, default=2000)
parser.add_argument("-t", "--test_episode", type=int, default=100)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-c", "--cpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
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

train_result = './train_result/'

label_list = ['Health', 'anomaly']


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
    loos_result = []
    loos_result_1 = []

    accuray_result_1_1 = []
    accuray_result_1_2 = []
    accuray_result_1_3 = []

    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    # metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders(train,test,label_list)

    # Step 2: init neural networks
    print("init neural networks")

    # 定义化网络
    feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    # relation_network = vit.ViT(image_size=28, patch_size=7, num_classes=2, dim=1024, depth=4, heads=8, mlp_dim=2048,
    #                            dropout=0.1, emb_dropout=0.1)  # 定义关系网络
    relation_network = KAN([28 * 28, 128, 8])
    kan = KAN([8 * 512, 512, 32, 2])
    # 关系网络和特征提取模块的加载话
    #    feature_encoder.apply(weights_init)
    #   relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    kan.cuda(GPU)

    # 优化器的定义
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)
    #
    # if os.path.exists(
    #         str("./models/feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
    #     feature_encoder.load_state_dict(torch.load(
    #         str("./models/feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
    #     print("load feature encoder success")
    # if os.path.exists(str("./models/relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     relation_network.load_state_dict(torch.load(str("./models/relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    #     print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0

    for episode in range(EPISODE):
        # print('episode', episode)
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0, 90, 180, 270])

        ##第一个监测点  轴箱motor
        metatrain_character_folders_1 = ['../CWT-1000/motor/train/health',
                                         '../CWT-1000/motor/train/anomaly']
        task_1 = tg.OmniglotTask(metatrain_character_folders_1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader_1 = tg.get_data_loader(task_1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                 shuffle=False, rotation=degrees)
        batch_dataloader_1 = tg.get_data_loader(task_1, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True,
                                                rotation=degrees)
        # sample datas
        sample_dataloader_1 = iter(sample_dataloader_1)
        samples_1, sample_labels_1 = next(sample_dataloader_1)

        # print('samples',samples.shape)
        batch_dataloader_1 = iter(batch_dataloader_1)
        batches_1, batch_labels_1 = next(batch_dataloader_1)


        ## 特征提取
        sample_features_1 = feature_encoder(Variable(samples_1).cuda(GPU))  # 5x64*5*5
        batch_features_1 = feature_encoder(Variable(batches_1).cuda(GPU))  # 20x64*5*5
        #######################################################################################


        ## 特征拼接
        # print(sample_features_1.shape)   # 2,128,28,28
        # print(sample_labels_1.shape)     # 2
        sample_features_ext_1 = sample_features_1.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        sample_labels_1 = sample_labels_1.repeat(BATCH_NUM_PER_CLASS)
        sample_labels_1 = sample_labels_1.long()
        batch_features_ext_1 = batch_features_1.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext_1 = torch.transpose(batch_features_ext_1, 0, 1)
        # print(batch_features_ext_1.shape)  # 38,2,128,28,28
        relation_pairs_1 = torch.cat((sample_features_ext_1, batch_features_ext_1), 2)
        # print(relation_pairs_1.shape)            #38,2,256,28,28
        # relation_pairs_1 = relation_pairs_1.view(-1, FEATURE_DIM * 2, 28, 28)
        relation_pairs_1 = relation_pairs_1.view(-1, FEATURE_DIM * 4, 28 * 28)


        ##计算关系分数 kan

        # print(relation_pairs_1.shape)

        relations_1 = relation_network(relation_pairs_1)
        relations_1 = relations_1.view(-1, 8 * 512)
        relations_1 = kan(relations_1)

        relations_1 = relations_1.view(-1, CLASS_NUM)
        # print(relations_1.shape)
        # print(relations_1.shape)
        ########################################################################################

        # ##计算关系分数 kan

        # # print(relation_pairs_1.shape)
        # relations_1 = relation_network(relation_pairs_1)
        # # print(relations_1.shape)
        # relations_1 = relations_1.view(38, 8 * 512)
        # relations_1 = kan(relations_1)
        # relations_1 = relations_1.view(-1, CLASS_NUM)
        # print(relations_1.shape)
        # print(relations_1.shape)
        # ########################################################################################
        # relations_2 = relation_network(relation_pairs_2)
        # relations_2 = relations_2.view(-1, CLASS_NUM)
        # ########################################################################################
        # relations_3 = relation_network(relation_pairs_3)
        # relations_3 = relations_3.view(-1, CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        # 计算LOSS
        batch_labels_1 = batch_labels_1.long()
        # print(batches_1.shape)
        # print(batch_labels_1.view(-1,1))
        one_hot_labels_1 = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1,
                                                                                            batch_labels_1.view(-1, 1),
                                                                                            1).cuda(GPU)
        # print(one_hot_labels_1.shape)
        # print(relations_1.shape)
        loss_1 = mse(relations_1, one_hot_labels_1)

        ########################################################################################


        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss = loss_1

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        loos_result_1.append(loss_1)

        loos_result.append(loss)

        if (episode + 1) % 50 == 0:
            print("episode:", episode + 1, "loss", loss.item())
            loos_result.append(loss)

        # if (episode + 1) % 100 == 0:
        #     # test
        #
        #     print("Testing...1-1,motor type1",end='')
        #     total_rewards_1_1 = 0
        #     for i in range(TEST_EPISODE):
        #         degrees = random.choice([0, 90, 180, 270])
        #         metatest_character_folders1 = ['../CWT-1000/motor/test/health',
        #                                        '../CWT-1000/motor/test/anomaly/anomalyTYPE1']
        #         # '../CWT-1000/motor/test/anomaly/anomalyTYPE13']
        #         # '../CWT-1000/motor/test/anomaly/anomalyTYPE14']
        #         metatrain_character_folders1 = ['../CWT-1000/motor/train/health',
        #                                         '../CWT-1000/motor/train/anomaly']
        #         task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
        #                                SAMPLE_NUM_PER_CLASS, )
        #         task1 = tg.OmniglotTask(metatrain_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
        #                                 BATCH_NUM_PER_CLASS)
        #         sample_dataloader = tg.get_data_loader(task1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
        #                                                shuffle=False, rotation=degrees)
        #         test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
        #                                              shuffle=True, rotation=degrees)
        #         sample_dataloader = iter(sample_dataloader)
        #         sample_images, sample_labels = next(sample_dataloader)
        #         test_dataloader = iter(test_dataloader)
        #         test_images, test_labels = next(test_dataloader)
        #         sample_features = feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
        #         test_features = feature_encoder(Variable(test_images).cuda(GPU))  # 20x64
        #         sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        #         test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        #         test_features_ext = torch.transpose(test_features_ext, 0, 1)
        #         # print(torch.cat((sample_features_ext, test_features_ext), 2).shape)        #2,2,256,28,28
        #         # relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 28,
        #         #                                                                              28)
        #         #
        #         relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1,
        #                                                                                      FEATURE_DIM * 4, 28 * 28)
        #         ##kan
        #         # relations1 = relation_network(relation_pairs)
        #         # # print(relations1.shape)
        #         # relations1 = relations1.view(2, 8 * 512)
        #         # relations1 = kan(relations1)
        #         # relations = relations1.view(-1, CLASS_NUM)
        #         #
        #         # # relations1 = relation_network(relation_pairs)
        #         # # relations = relations1.view(-1, CLASS_NUM)
        #         # bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)
        #
        #         # transformer
        #         relations1 = relation_network(relation_pairs)
        #         relations1 = relations1.view(2, 8 * 512)
        #         relations1 = kan(relations1)
        #         relations = relations1.view(-1, CLASS_NUM)
        #         bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)
        #
        #         # print(relations)
        #         for j in range(len(relations)):
        #             if relations[j][0] > 0.9:
        #                 bb[j] = 0
        #             else:
        #                 bb[j] = 1
        #         # _,predict_labels = torch.max(relations.data,1)
        #         predict_labels = bb.cpu()
        #         rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
        #         total_rewards_1_1 += np.sum(rewards)
        #     test_accuracy_1_1 = total_rewards_1_1 / 1.0 / CLASS_NUM / TEST_EPISODE
        #     accuray_result_1_1.append(test_accuracy_1_1)
        #     print(" test accuracy 1-1:", test_accuracy_1_1)
        #
        #     print("Testing...1-2,motor type2", end='')
        #     total_rewards_1_2 = 0
        #     for i in range(TEST_EPISODE):
        #         degrees = random.choice([0, 90, 180, 270])
        #         metatest_character_folders1 = ['../CWT-1000/motor/test/health',
        #                                        '../CWT-1000/motor/test/anomaly/anomalyTYPE2']
        #         # '../CWT-1000/motor/test/anomaly/anomalyTYPE13']
        #         # '../CWT-1000/motor/test/anomaly/anomalyTYPE14']
        #         metatrain_character_folders1 = ['../CWT-1000/motor/train/health',
        #                                         '../CWT-1000/motor/train/anomaly']
        #         task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
        #                                SAMPLE_NUM_PER_CLASS, )
        #         task1 = tg.OmniglotTask(metatrain_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
        #                                 BATCH_NUM_PER_CLASS)
        #         sample_dataloader = tg.get_data_loader(task1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
        #                                                shuffle=False, rotation=degrees)
        #         test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
        #                                              shuffle=True, rotation=degrees)
        #         sample_dataloader = iter(sample_dataloader)
        #         sample_images, sample_labels = next(sample_dataloader)
        #         test_dataloader = iter(test_dataloader)
        #         test_images, test_labels = next(test_dataloader)
        #         sample_features = feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
        #         test_features = feature_encoder(Variable(test_images).cuda(GPU))  # 20x64
        #         sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        #         test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        #         test_features_ext = torch.transpose(test_features_ext, 0, 1)
        #         # print(torch.cat((sample_features_ext, test_features_ext), 2).shape)        #2,2,256,28,28
        #         # relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 28,
        #         #                                                                              28)
        #         #
        #         relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1,
        #                                                                                      FEATURE_DIM * 4, 28 * 28)
        #         ##kan
        #         # relations1 = relation_network(relation_pairs)
        #         # # print(relations1.shape)
        #         # relations1 = relations1.view(2, 8 * 512)
        #         # relations1 = kan(relations1)
        #         # relations = relations1.view(-1, CLASS_NUM)
        #         #
        #         # # relations1 = relation_network(relation_pairs)
        #         # # relations = relations1.view(-1, CLASS_NUM)
        #         # bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)
        #
        #         # transformer
        #         relations1 = relation_network(relation_pairs)
        #         relations1 = relations1.view(2, 8 * 512)
        #         relations1 = kan(relations1)
        #         relations = relations1.view(-1, CLASS_NUM)
        #         bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)
        #
        #         # print(relations)
        #         for j in range(len(relations)):
        #             if relations[j][0] > 0.9:
        #                 bb[j] = 0
        #             else:
        #                 bb[j] = 1
        #         # _,predict_labels = torch.max(relations.data,1)
        #         predict_labels = bb.cpu()
        #         rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
        #         total_rewards_1_2 += np.sum(rewards)
        #     test_accuracy_1_2 = total_rewards_1_2 / 1.0 / CLASS_NUM / TEST_EPISODE
        #     accuray_result_1_2.append(test_accuracy_1_2)
        #     print(" test accuracy 1-2:", test_accuracy_1_2)
        #
        #     print("Testing...1-3,motor type3", end='')
        #     total_rewards_1_3 = 0
        #     for i in range(TEST_EPISODE):
        #         degrees = random.choice([0, 90, 180, 270])
        #         metatest_character_folders1 = ['../CWT-1000/motor/test/health',
        #                                        '../CWT-1000/motor/test/anomaly/anomalyTYPE3']
        #         # '../CWT-1000/motor/test/anomaly/anomalyTYPE13']
        #         # '../CWT-1000/motor/test/anomaly/anomalyTYPE14']
        #         metatrain_character_folders1 = ['../CWT-1000/motor/train/health',
        #                                         '../CWT-1000/motor/train/anomaly']
        #         task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
        #                                SAMPLE_NUM_PER_CLASS, )
        #         task1 = tg.OmniglotTask(metatrain_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
        #                                 BATCH_NUM_PER_CLASS)
        #         sample_dataloader = tg.get_data_loader(task1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
        #                                                shuffle=False, rotation=degrees)
        #         test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
        #                                              shuffle=True, rotation=degrees)
        #         sample_dataloader = iter(sample_dataloader)
        #         sample_images, sample_labels = next(sample_dataloader)
        #         test_dataloader = iter(test_dataloader)
        #         test_images, test_labels = next(test_dataloader)
        #         sample_features = feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
        #         test_features = feature_encoder(Variable(test_images).cuda(GPU))  # 20x64
        #         sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        #         test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        #         test_features_ext = torch.transpose(test_features_ext, 0, 1)
        #         # print(torch.cat((sample_features_ext, test_features_ext), 2).shape)        #2,2,256,28,28
        #         # relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 28,
        #         #                                                                              28)
        #         #
        #         relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1,
        #                                                                                      FEATURE_DIM * 4, 28 * 28)
        #         ##kan
        #         # relations1 = relation_network(relation_pairs)
        #         # # print(relations1.shape)
        #         # relations1 = relations1.view(2, 8 * 512)
        #         # relations1 = kan(relations1)
        #         # relations = relations1.view(-1, CLASS_NUM)
        #         #
        #         # # relations1 = relation_network(relation_pairs)
        #         # # relations = relations1.view(-1, CLASS_NUM)
        #         # bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)
        #
        #         # transformer
        #         relations1 = relation_network(relation_pairs)
        #         relations1 = relations1.view(2, 8 * 512)
        #         relations1 = kan(relations1)
        #         relations = relations1.view(-1, CLASS_NUM)
        #         bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)
        #
        #         # print(relations)
        #         for j in range(len(relations)):
        #             if relations[j][0] > 0.9:
        #                 bb[j] = 0
        #             else:
        #                 bb[j] = 1
        #         # _,predict_labels = torch.max(relations.data,1)
        #         predict_labels = bb.cpu()
        #         rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
        #         total_rewards_1_3 += np.sum(rewards)
        #     test_accuracy_1_3 = total_rewards_1_3 / 1.0 / CLASS_NUM / TEST_EPISODE
        #     accuray_result_1_3.append(test_accuracy_1_3)
        #     print(" test accuracy 1-3:", test_accuracy_1_3)
        #     #############################################################################################################
        #
        #
        #     ###总精度
        #     test_accuracy = test_accuracy_1_1 + test_accuracy_1_2 + test_accuracy_1_3
        #
        #     test_accuracy = test_accuracy / 3.0
        #     print("TOtal_ accuracy:", test_accuracy)
        #
        #     # if test_accuracy >= last_accuracy:
        #     #     # save networks
        #     #     torch.save(feature_encoder.state_dict(),
        #     #                str("./models/motor_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
        #     #                    SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
        #     #     torch.save(relation_network.state_dict(),
        #     #                str("./models/motor_relation_network_" + str(CLASS_NUM) + "way_" + str(
        #     #                    SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
        #     #     torch.save(kan.state_dict(),
        #     #                str("./models/motor_relation_network_2" + str(CLASS_NUM) + "way_" + str(
        #     #                    SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
        #     #
        #     #     print("save networks for episode:", episode)
        #     #
        #     #     last_accuracy = test_accuracy

    return loos_result_1


if __name__ == '__main__':
    loos_result_1 = main()
    loos_result_1_cpu = [x_1.cpu().detach().numpy() for x_1 in loos_result_1]

    np.savetxt(train_result + 'motor2000_train_loss_1.csv', loos_result_1_cpu, fmt='%.8f', delimiter=',')


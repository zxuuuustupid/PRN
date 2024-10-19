#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


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


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 128)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default =2)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 5)
parser.add_argument("-t","--test_episode", type = int, default = 2000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
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

#填写测试负载






test_result='./test_result/'


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

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

    feature_encoder = CNNEncoder1.rsnet()  #特征提取
    relation_network =  vit.ViT(image_size=28,patch_size=7,num_classes=2,dim=1024,depth=4,heads=8,mlp_dim =2048,dropout=0.1,emb_dropout=0.1 )   #定义关系网络


    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)


    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)   
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")




    total_accuracy = 0.0
    for episode in range(1):

        ture_result=[]
        predict_result=[]
        t_labels=[]
        scores_result=[]
        h=[]
        # test
        print("Testing...")
        total_rewards = 0
        accuracies = []
        with torch.no_grad():
            for i in range(TEST_EPISODE):
                degrees = random.choice([0,90,180,270])
                
                #metatest_character_folders1=['../similar_socre/Health','../similar_socre/anomaly']
                metatest_character_folders1=['../nosie_data/test_2/Health','../nosie_data/test_2/anomaly'] #
                #metatest_character_folders1=['../nosie_8/test/Health','../nosie_8/test/anomaly']
                metatrain_character_folders1=['../train_data_Bearin2/train/Health','../train_data_Bearin2/train/anomaly']
                
                task = tg.OmniglotTask(metatest_character_folders1,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS,)
                task1 = tg.OmniglotTask(metatrain_character_folders1,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS,)

                sample_dataloader = tg.get_data_loader(task1,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
                test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

                sample_images,sample_l0abels = sample_dataloader.__iter__().next()
                
                test_images,test_labels = test_dataloader.__iter__().next()
                #print('test_labels',test_labels)

                # calculate features
                sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64
                
                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                test_features_ext = torch.transpose(test_features_ext,0,1)

                relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,28,28)
                relations1 = relation_network(relation_pairs)
                relations=relations1.view(-1,CLASS_NUM)
                bb=Variable(torch.zeros( CLASS_NUM)).cuda(GPU)
                
                

                
               # print(relations)
                
                for j in range(len(relations)):
                   
                    if relations[j][0] >=0.95 :
                        bb[j]=0
                         
                        scores_result.append(relations[j][0].cpu().item()) #保存相似分数
                        predict_result.append(0) #保存预测结果
                        ture_result.append(test_labels[j].cpu().item())#保存真实结果
                    else:
                        bb[j]=1
                        scores_result.append(relations[j][0].cpu().item()) #保存相似分数
                        predict_result.append(1) #保存预测结果
                        ture_result.append(test_labels[j].cpu().item())#保存真实结果
                        
                       

#                _,predict_labels = torch.max(relations.data,1)
                predict_labels=bb.cpu()


                #_,predict_labels = torch.max(relations.data,1)
                
                rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM)]
                #print(len(rewards))

                total_rewards += np.sum(rewards)
                accuracy = np.sum(rewards)/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS
                accuracies.append(accuracy)
               
                

  
            pass

        test_accuracy,h = mean_confidence_interval(accuracies)

        
       


        print("test accuracy:",test_accuracy,"h:",h)
        total_accuracy += test_accuracy

        print("aver_accuracy:",total_accuracy/EPISODE)
        





if __name__ == '__main__':
    main()

import os
import csv
import yaml
import torch
import torch.nn as nn
import argparse
import wandb
import random
from tqdm import tqdm
import logging
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report, average_precision_score
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model.IIDLPepPI import IIDLPepPI, IIDLPepPIRes
from generate_peptide_features import PepFeature
from generate_protein_features import ProFeature
from torch.utils.data import Dataset, DataLoader
from datapro import convert


#数据集类

#冻结特定层，
def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    # 解冻输出层以进行微调
    for param in model.output.parameters():
        param.requires_grad = True
    for param in model.dnns.parameters():
        param.requires_grad = True
#替换最后分类层
def replace_output_layer(model):
    # 替换最后一层分类器
    model.dnns = nn.Sequential(
            nn.Linear(850,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024))
    model.output = nn.Linear(1024, 850)
    return model


#ROC AUC分数和平均精度分数
def cls_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # r2_score, mean_squred_error are ignored
    return roc_auc_score(label, pred), average_precision_score(label, pred)

def random_split(X, Y, fold=5):
    skf = StratifiedKFold(n_splits=fold, shuffle=True)
    train_idx_list, test_idx_list = [], []
    for train_index, test_index in skf.split(X, Y):
        train_idx_list.append(train_index)
        test_idx_list.append(test_index)
    return train_idx_list, test_idx_list

#加载模型
def load_checkpoint(filepath, args):
    ckpt = torch.load(filepath)
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad=False
    model.eval()
    return model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)

#数据批处理
def collate_fn(batch):
    x_pep = torch.tensor([f["x_pep"] for f in batch])
    x_2_pep = torch.tensor([f["x_2_pep"] for f in batch])
    # x_dense_pep = [f["x_dense_pep"] for f in batch]
    # x_bert_pep = [f["x_bert_pep"] for f in batch]
    x_p = torch.tensor([f["x_p"] for f in batch])
    x_2_p = torch.tensor([f["x_2_p"] for f in batch])
    # x_dense_p = [f["x_dense_p"] for f in batch]
    # x_bert_p = [f["x_bert_p"] for f in batch]
    label = torch.tensor([f["label"] for f in batch])
    # res_label = torch.tensor([f["res_label"] for f in batch])
    pep_res_label = torch.tensor([f["pep_res_label"] for f in batch])
    pro_res_label = torch.tensor([f["pro_res_label"] for f in batch])
    res_len = torch.tensor([f["res_len"] for f in batch])
    #output = (x_pep, x_2_pep, x_dense_pep, x_bert_pep, x_p, x_2_p, x_dense_p, x_bert_p, label, res_label)
    output = (x_pep, x_2_pep, x_p, x_2_p, label, pep_res_label, pro_res_label, res_len)
    return output

def train(args, model, train_feature, device, criterion, optimizer, nums_step):
    preds, labels = [], []
    avg_loss = 0
    criterion.to(device)
    model.train()
    # print('start loading')
    for step, batch in enumerate(tqdm(train_feature)):
        model.zero_grad()
        inputs = {'x_pep': batch[0].to(args.device),
                  'x_2_pep': batch[1].to(args.device),
                  # 'x_dense_pep': batch[2].to(args.device),
                  # 'x_bert_pep': batch[3].to(args.device),
                  'x_p': batch[2].to(args.device),
                  'x_2_p': batch[3].to(args.device),
                  # 'x_dense_p': batch[6].to(args.device),
                  # 'x_bert_p': batch[7].to(args.device),
                  }
        pred = model(**inputs)
        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(batch[4].detach().cpu().numpy().tolist())
        labels_float = batch[4].to(torch.float32).to(args.device)
        #print(pred.device, labels_float.device)
        loss = criterion(pred, labels_float)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        nums_step += 1
        wandb.log({"loss": loss.item()}, step=nums_step)

    preds = np.array(preds)
    labels = np.array(labels)
    #准确率
    acc = accuracy_score(labels, np.round(preds))
    #AUROC分数、AUPR分数
    test_scores = cls_scores(labels, preds)
    AUC = round(test_scores[0], 6)
    AUPR = round(test_scores[1], 6)
    avg_loss /= len(train_feature)

    return avg_loss, acc, AUC, AUPR, nums_step

def res_train(args, model, train_feature, device, criterion, optimizer, nums_step):
    preds_pep, preds_pro, pep_labels, pro_labels, res_len = [], [], [], [], []
    avg_loss = 0
    criterion.to(device)
    model.train()
    # print('start loading')
    for step, batch in enumerate(tqdm(train_feature)):
        model.zero_grad()
        inputs = {'x_pep': batch[0].to(args.device),
                  'x_2_pep': batch[1].to(args.device),
                  # 'x_dense_pep': batch[2].to(args.device),
                  # 'x_bert_pep': batch[3].to(args.device),
                  'x_p': batch[2].to(args.device),
                  'x_2_p': batch[3].to(args.device),
                  # 'x_dense_p': batch[6].to(args.device),
                  # 'x_bert_p': batch[7].to(args.device),
                  }
        pred = model(**inputs) #[bs,850]
        #要对pred进行划分，分为pep和pro的
        # Binding residues prediction result
        peptide_residue = pred[0:, 0:50]
        protein_residue = pred[0:, 50:850]
        preds_pep.extend(peptide_residue.detach().cpu().numpy().tolist())
        preds_pro.extend(protein_residue.detach().cpu().numpy().tolist())
        res_len.extend(batch[7].detach().cpu().numpy().tolist())
        pep_labels.extend(batch[5].detach().cpu().numpy().tolist())
        pro_labels.extend(batch[6].detach().cpu().numpy().tolist())
        pep_labels_float = batch[5].to(torch.float32).to(args.device)
        pro_labels_float = batch[6].to(torch.float32).to(args.device)

        # 展平
        peptide_residue_flat = peptide_residue.reshape(-1)
        pep_labels_float_flat = pep_labels_float.reshape(-1)
        pep_loss = criterion(peptide_residue_flat, pep_labels_float_flat)

        pro_residue_flat = protein_residue.reshape(-1)
        pro_labels_float_flat = pro_labels_float.reshape(-1)
        pro_loss = criterion(pro_residue_flat, pro_labels_float_flat)

        total_loss = pep_loss + pro_loss
        total_loss.backward()
        optimizer.step()
        avg_loss += total_loss.item()
        nums_step += 1
        wandb.log({"pep_loss": pep_loss.item()}, step=nums_step)
        wandb.log({"pro_loss": pro_loss.item()}, step=nums_step)

    #计算肽侧评价指标
    preds_pep = np.array(preds_pep)
    pep_labels = np.array(pep_labels)
    res_lengths = np.array(res_len)
    # 初始化展平后的数组
    flat_preds = []
    flat_labels = []

    # 遍历每个肽-蛋白对
    for i in range(len(pep_labels)):
        # 获取该肽的实际长度
        length = res_lengths[i, 0]
        if length < 50:
            # 只取前length个预测值和标签
            flat_preds.extend(preds_pep[i, :length])
            flat_labels.extend(pep_labels[i, :length])
        else:
            flat_preds.extend(preds_pep[i, :])
            flat_labels.extend(pep_labels[i, :])

    # 转换为NumPy数组
    flat_preds = np.array(flat_preds)
    flat_labels = np.array(flat_labels)
    pep_acc = accuracy_score(flat_labels, np.round(flat_preds))
    test_scores = cls_scores(flat_labels, flat_preds)
    pep_AUC = round(test_scores[0], 6)
    pep_AUPR = round(test_scores[1], 6)

    # 计算蛋白质侧评价指标
    preds_pro = np.array(preds_pro)
    pro_labels = np.array(pro_labels)
    # 初始化展平后的数组
    flat_preds = []
    flat_labels = []

    # 遍历每个肽-蛋白对
    for i in range(len(pro_labels)):
        # 获取该肽的实际长度
        length = res_lengths[i][0]
        if length < 800:
            # 只取前length个预测值和标签
            flat_preds.extend(preds_pro[i, :length])
            flat_labels.extend(pro_labels[i, :length])
        else:
            flat_preds.extend(preds_pro[i, :])
            flat_labels.extend(pro_labels[i, :])

    # 转换为NumPy数组
    flat_preds = np.array(flat_preds)
    flat_labels = np.array(flat_labels)
    pro_acc = accuracy_score(flat_labels, np.round(flat_preds))
    test_scores = cls_scores(flat_labels, np.round(flat_preds))
    pro_AUC = round(test_scores[0], 6)
    pro_AUPR = round(test_scores[1], 6)
    # pep_acc = accuracy_score(pep_labels, np.round(preds_pep))
    # test_scores = cls_scores(pep_labels, preds_pep)
    # pep_AUC = round(test_scores[0], 6)
    # pep_AUPR = round(test_scores[1], 6)
    #
    # #计算蛋白质侧评价指标
    # preds_pro = np.array(preds_pro)
    # pro_labels = np.array(pro_labels)
    # pro_acc = accuracy_score(pro_labels, np.round(preds_pro))
    # test_scores = cls_scores(pro_labels, preds_pro)
    # pro_AUC = round(test_scores[0], 6)
    # pro_AUPR = round(test_scores[1], 6)

    avg_loss /= len(train_feature)

    return avg_loss, pep_acc, pep_AUC, pep_AUPR, pro_acc, pro_AUC, pro_AUPR, nums_step

def test(args, model, dev_feature, device, criterion):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        avg_loss = 0
        for step, batch in enumerate(dev_feature):
            inputs = {'x_pep': batch[0].to(args.device),
                      'x_2_pep': batch[1].to(args.device),
                      # 'x_dense_pep': batch[2].to(args.device),
                      # 'x_bert_pep': batch[3].to(args.device),
                      'x_p': batch[2].to(args.device),
                      'x_2_p': batch[3].to(args.device),
                      # 'x_dense_p': batch[6].to(args.device),
                      # 'x_bert_p': batch[7].to(args.device),
                      }

            pred = model(**inputs)
            preds.extend(pred.detach().cpu().numpy().tolist())
            labels.extend(batch[4].detach().cpu().numpy().tolist())
            labels_float = batch[4].to(torch.float32).to(args.device)
            loss = criterion(pred, labels_float)
            avg_loss += loss.item()

        avg_loss /= len(dev_feature)

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, np.round(preds))
    F1 = f1_score(labels, np.round(preds))
    # AUROC分数、AUPR分数、F1分数
    test_scores = cls_scores(labels, preds)
    AUC = round(test_scores[0], 6)
    AUPR = round(test_scores[1], 6)
    #ROC,PR曲线
    fpr, tpr, thresholds = roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    avg_loss /= len(dev_feature)

    return avg_loss, acc, AUC, AUPR, F1, fpr, tpr, precision, recall

def res_test(args, model, dataloader, device, criterion):
    model.eval()
    preds_pep, preds_pro, pep_labels, pro_labels, res_len = [], [], [], [], []
    with torch.no_grad():
        avg_loss = 0
        for step, batch in enumerate(dataloader):
            inputs = {'x_pep': batch[0].to(args.device),
                      'x_2_pep': batch[1].to(args.device),
                      # 'x_dense_pep': batch[2].to(args.device),
                      # 'x_bert_pep': batch[3].to(args.device),
                      'x_p': batch[2].to(args.device),
                      'x_2_p': batch[3].to(args.device),
                      # 'x_dense_p': batch[6].to(args.device),
                      # 'x_bert_p': batch[7].to(args.device),
                      }

            pred = model(**inputs)
            peptide_residue = pred[0:, 0:50] #[bs, 50]
            protein_residue = pred[0:, 50:850] #[bs, 800]
            preds_pep.extend(peptide_residue.detach().cpu().numpy().tolist())#嵌套列表
            preds_pro.extend(protein_residue.detach().cpu().numpy().tolist())
            res_len.extend(batch[7].detach().cpu().numpy().tolist())
            pep_labels.extend(batch[5].detach().cpu().numpy().tolist())
            pro_labels.extend(batch[6].detach().cpu().numpy().tolist())
            pep_labels_float = batch[5].to(torch.float32).to(args.device) #[bs,50]
            pro_labels_float = batch[6].to(torch.float32).to(args.device) #[bs, 800]
            #展平
            peptide_residue_flat = peptide_residue.reshape(-1)
            pep_labels_float_flat = pep_labels_float.reshape(-1)
            pep_loss = criterion(peptide_residue_flat, pep_labels_float_flat)

            pro_residue_flat = protein_residue.reshape(-1)
            pro_labels_float_flat = pro_labels_float.reshape(-1)
            pro_loss = criterion(pro_residue_flat, pro_labels_float_flat)
            total_loss = pep_loss + pro_loss
            avg_loss += total_loss.item()

        avg_loss /= len(dataloader)

    # 计算肽侧评价指标
    preds_pep = np.array(preds_pep)
    pep_labels = np.array(pep_labels)
    res_lengths = np.array(res_len)
    # 初始化展平后的数组
    flat_preds = []
    flat_labels = []

    # 遍历每个肽-蛋白对
    for i in range(len(pep_labels)):
        # 获取该肽的实际长度
        length = res_lengths[i,0]
        if length < 50:
            # 只取前length个预测值和标签
            flat_preds.extend(preds_pep[i, :length])
            flat_labels.extend(pep_labels[i, :length])
        else:
            flat_preds.extend(preds_pep[i, :])
            flat_labels.extend(pep_labels[i, :])

    # 转换为NumPy数组
    flat_preds = np.array(flat_preds)
    flat_labels = np.array(flat_labels)
    pep_F1 = f1_score(flat_labels, np.round(flat_preds))
    pep_acc = accuracy_score(flat_labels, np.round(flat_preds))
    test_scores = cls_scores(flat_labels, flat_preds)
    pep_AUC = round(test_scores[0], 6)
    pep_AUPR = round(test_scores[1], 6)
    # roc, PR曲线
    pep_fpr, pep_tpr, _ = roc_curve(flat_labels, flat_preds)
    pep_precision, pep_recall, _ = precision_recall_curve(flat_labels, flat_preds)

    # 计算蛋白质侧评价指标
    preds_pro = np.array(preds_pro)
    pro_labels = np.array(pro_labels)
    # 初始化展平后的数组
    flat_preds = []
    flat_labels = []

    # 遍历每个肽-蛋白对
    for i in range(len(pro_labels)):
        # 获取该肽的实际长度
        length = res_lengths[i][0]
        if length < 800:
            # 只取前length个预测值和标签
            flat_preds.extend(preds_pro[i, :length])
            flat_labels.extend(pro_labels[i, :length])
        else:
            flat_preds.extend(preds_pro[i, :])
            flat_labels.extend(pro_labels[i, :])

    # 转换为NumPy数组
    flat_preds = np.array(flat_preds)
    flat_labels = np.array(flat_labels)
    pro_F1 = f1_score(flat_labels, np.round(flat_preds))
    pro_acc = accuracy_score(flat_labels, np.round(flat_preds))
    test_scores = cls_scores(flat_labels, np.round(flat_preds))
    pro_AUC = round(test_scores[0], 6)
    pro_AUPR = round(test_scores[1], 6)
    #roc, PR曲线
    pro_fpr, pro_tpr, _ = roc_curve(flat_labels, flat_preds)
    pro_precision, pro_recall, _ = precision_recall_curve(flat_labels, flat_preds)

    # 返回所有性能指标
    results = {
        'pep': {
            'F1': pep_F1,
            'accuracy': pep_acc,
            'AUC': pep_AUC,
            'AUPR': pep_AUPR,
            'ROC': {'fpr': pep_fpr, 'tpr': pep_tpr},
            'PR': {'precision': pep_precision, 'recall': pep_recall}
        },
        'pro': {
            'F1': pro_F1,
            'accuracy': pro_acc,
            'AUC': pro_AUC,
            'AUPR': pro_AUPR,
            'ROC': {'fpr': pro_fpr, 'tpr': pro_tpr},
            'PR': {'precision': pro_precision, 'recall': pro_recall}
        }
    }

    return results

    #return avg_loss, acc, AUC, AUPR, F1, fpr, tpr, precision, recall


def main():
    # 定义超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="res", type=str)
    parser.add_argument("--data_dir", default="./datasets/5-fold Randomly Split Datasets", type=str,
                        help="The input data dir.")
    parser.add_argument("--model_dir", default="", type=str,
                        help="The model dir.")
    parser.add_argument("--train_seq_file", default="", type=str,
                        help="The train seq file.")
    parser.add_argument("--dev_seq_file", default="", type=str)
    parser.add_argument("--test_seq_file", default="", type=str)
    parser.add_argument("--train_res_file", default="", type=str,
                        help="The train res file.")
    parser.add_argument("--dev_res_file", default="", type=str)
    parser.add_argument("--test_res_file", default="", type=str)
    parser.add_argument("--seq_saved_path",default="./ckpts/seq_model_best_F1.pkl", type=str,)
    parser.add_argument("--res_saved_path", default="./ckpts/res_model_best_F1.pkl", type=str,)
    parser.add_argument("--seq_load_path", default="", type=str, )
    parser.add_argument("--res_load_path", default="", type=str,
                        help="The saved model.")
    parser.add_argument("--num_train_epochs", default=50.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=16, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--n_lr", default=1e-4, type=float,
                        help="The new layer learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    #处理数据
    prepro = convert

    train_feature = prepro(args.data_dir,'train', args.task_name)
    dev_feature = prepro(args.data_dir, 'dev', args.task_name)
    test_feature = prepro(args.data_dir, 'test', args.task_name)
    # 得到的特征x_p等长度和原始数据不一致，会有重复蛋白质序列和肽序列。对于每一对，需要根据初始蛋白质序列和肽序列的索引进行特征提取

    set_seed(args)
    # EPOCHS = 100
    criterion = nn.BCELoss()
    if args.task_name == "seq":
        #序列级任务
        #蛋白质特征
        if args.seq_load_path == "":
            #5折交叉训练
            n_fold = 5
            wandb.init(project="yjy_PepPI_0807")
            # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            dev_AUC_list, dev_AUPR_list, dev_F1_list = [], [], []
            train_step = 0
            best_score = -1  #保存最好的模型
            for fold in range(n_fold):
                train_feature_cur = train_feature[fold] #当前折的训练数据
                # train_feature_cur = [item for i, sublist in enumerate(train_feature) if i != fold for item in sublist]
                dev_feature_cur = dev_feature[fold]
                # 数据批处理
                train_dataloader = DataLoader(train_feature_cur, batch_size=args.train_batch_size, shuffle=True,
                                              collate_fn=collate_fn, drop_last=True)
                dev_dataloader = DataLoader(dev_feature_cur, batch_size=args.test_batch_size, shuffle=False,
                                            collate_fn=collate_fn, drop_last=True)
                model = IIDLPepPI()
                model = model.to(device)

                print("Start seq training and device in use:", device)
                #optimizer = optim.Adam(model.parameters(), lr=1e-4)

                optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
                #optimizer = optim.RMSprop(model.parameters(), lr=0.0005) #weight_decay=1e-6
                for e in range(int(args.num_train_epochs)):
                    print(f"Epoch:{e + 1}")
                    train_loss, train_acc, train_AUC, train_AUPR, nums_step = train(args, model, train_dataloader, device, criterion, optimizer,train_step)
                    #评估模型
                    dev_loss, dev_acc, dev_AUC, dev_AUPR, dev_f1, *_ = test(args, model, dev_dataloader, device, criterion)
                    train_step = nums_step
                    # print(f"Training loss:{train_loss:.4f}, ACC:{train_acc:.4f}, AUC:{train_auc:.4f}, cls AUC:{train_AUC:.6f}, cls AUPR:{train_AUPR:.6f}")
                    # print(f"Test loss:{test_loss:.4f}, ACC:{test_acc:.4f}, AUC:{test_auc:.4f}, cls AUC:{test_AUC:.6f}, cls AUPR:{test_AUPR:.6f}")
                    print(f"Training loss:{train_loss:.8f}, ACC:{train_acc:.4f}, AUC:{train_AUC:.6f}, AUPR:{train_AUPR:.6f}")
                    print(f"Dev ACC:{dev_acc:.4f}, AUC:{dev_AUC:.6f}, AUPR:{dev_AUPR:.6f}, F1:{dev_f1:.6f}")
                    if dev_f1 >= best_score:
                        best_auc = dev_AUC
                        best_aupr = dev_AUPR
                        best_score = dev_f1
                        print(f'Save model of epoch {e} with {n_fold}-fold cv')
                        checkpoint = {'model': IIDLPepPI(), 'model_state_dict': model.state_dict()}
                        torch.save(checkpoint, args.seq_saved_path)
                dev_AUC_list.append(best_auc)
                dev_AUPR_list.append(best_aupr)
                dev_F1_list.append(best_score)
                #保留最后一个模型

            print('fold mean auc & aupr & f1', np.mean(dev_AUC_list, axis=0), np.mean(dev_AUPR_list, axis=0), np.mean(dev_F1_list, axis=0))
            print('fold std auc & aupr & f1', np.std(dev_AUC_list, axis=0), np.std(dev_AUPR_list, axis=0), np.std(dev_F1_list, axis=0))
            mean_auc = np.mean(dev_AUC_list, axis=0)
            mean_aupr = np.mean(dev_AUPR_list, axis=0)
            mean_f1 = np.mean(dev_F1_list, axis=0)
            std_auc = np.std(dev_AUC_list, axis=0)
            std_aupr = np.std(dev_AUPR_list, axis=0)
            std_f1 = np.std(dev_F1_list, axis=0)
            #wandb记录
            wandb.log({'mean_auc': mean_auc,
                       'mean_aupr': mean_aupr,
                       'mean_f1': mean_f1,
                       'std_auc': std_auc,
                       'std_aupr': std_aupr,
                       'std_f1': std_f1})
            wandb.finish()
        else:
            test_dataloader = DataLoader(test_feature, batch_size=args.test_batch_size, shuffle=False,
                                        collate_fn=collate_fn, drop_last=True)
            model_ckpt = load_checkpoint(args.seq_load_path, args)
            model_ckpt = model_ckpt.to(device)
            #测试集绘制曲线
            test_loss, test_acc, test_AUC, test_AUPR, test_f1, test_fpr, test_tpr, test_precision, test_recall = test(args, model_ckpt, test_dataloader, device, criterion)
            print(f"Test ACC:{test_acc:.4f}, AUC:{test_AUC:.6f}, AUPR:{test_AUPR:.6f}, F1:{test_f1:.6f}")
            #绘制ROC和PR曲线
            plt.figure()
            plt.plot(test_fpr, test_tpr, label='ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic of seq')
            plt.legend(loc="lower right")
            plt.show()

            # 绘制Precision-Recall曲线
            plt.figure()
            plt.plot(test_recall, test_precision, label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0, None])
            plt.title('Precision-Recall Curve of seq')
            plt.legend(loc="upper right")
            plt.show()

    else:
        #残基级任务
        if args.res_load_path == "":
            #5折交叉训练
            n_fold = 5
            wandb.init(project="yjy_PepPI_res_0807")
            # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            dev_pep_AUC_list, dev_pep_AUPR_list, dev_pep_F1_list = [], [], []
            dev_pro_AUC_list, dev_pro_AUPR_list, dev_pro_F1_list = [], [], []
            train_step = 0
            best_score = -1
            for fold in range(n_fold):
                #取得是肽-蛋白的正样本对
                train_feature_cur = train_feature[fold] #当前折的训练数据
                dev_feature_cur = dev_feature[fold]
                # 数据批处理
                train_dataloader = DataLoader(train_feature_cur, batch_size=args.train_batch_size, shuffle=True,
                                              collate_fn=collate_fn, drop_last=True)
                dev_dataloader = DataLoader(dev_feature_cur, batch_size=args.test_batch_size, shuffle=False,
                                            collate_fn=collate_fn, drop_last=True)
                #加载序列级最佳模型参数

                res_model = IIDLPepPIRes()
                model = res_model.to(device)
                ckpt = torch.load(args.seq_saved_path)
                #加载序列级除dnns,output层的所有参数
                shared_params = {}
                # 排除不需要迁移的层
                exclude_layers = ['dnns.0.weight','dnns.0.bias','dnns.3.weight',
                                 'dnns.3.bias','dnns.6.weight','dnns.6.bias', 'output.weight', 'output.bias']
                # 复制需要迁移层的参数
                for key, value in ckpt['model_state_dict'].items():
                     if not any(exclude_layer == key for exclude_layer in exclude_layers):
                         shared_params[key] = value

                model.load_state_dict(shared_params, strict=False)
                #model = replace_output_layer(model)
                #freeze_layers(model)
                
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                # dev_loader = DataLoader(dataset=dev_dataset, batch_size=128, shuffle=True)
                print("Start training on res level:", device)
                # optimizer = optim.RMSprop(model.parameters(),lr=0.0005) #weight_decay=1e-6
                for e in range(int(args.num_train_epochs)):
                    print(f"Epoch:{e + 1}")
                    train_loss, train_pep_acc, train_pep_AUC, train_pep_AUPR, train_pro_acc, train_pro_AUC, train_pro_AUPR, nums_step = res_train(args, model, train_dataloader, device, criterion, optimizer,train_step)
                    #评估模型
                    # dev_loss, dev_acc, dev_AUC, dev_AUPR, dev_f1, *_ = res_test(args, model, dev_dataloader, device, criterion)
                    result = res_test(args, model, dev_dataloader, device, criterion)
                    train_step = nums_step
                    # print(f"Training loss:{train_loss:.4f}, ACC:{train_acc:.4f}, AUC:{train_auc:.4f}, cls AUC:{train_AUC:.6f}, cls AUPR:{train_AUPR:.6f}")
                    # print(f"Test loss:{test_loss:.4f}, ACC:{test_acc:.4f}, AUC:{test_auc:.4f}, cls AUC:{test_AUC:.6f}, cls AUPR:{test_AUPR:.6f}")
                    print(f"Training loss: {train_loss:.4f}, Peptide ACC: {train_pep_acc:.4f}, Peptide AUC: {train_pep_AUC:.6f}, Peptide AUPR: {train_pep_AUPR:.6f}")
                    print(f"Training loss: {train_loss:.4f}, Protein ACC: {train_pro_acc:.4f}, Protein AUC: {train_pro_AUC:.6f}, Protein AUPR: {train_pro_AUPR:.6f}")
                    # print(f"Training loss:{train_loss:.4f}, ACC:{train_acc:.4f}, AUC:{train_AUC:.6f}, AUPR:{train_AUPR:.6f}")
                    print(f"Peptide ACC:{result['pep']['accuracy']:.4f}, AUC:{result['pep']['AUC']:.6f}, AUPR:{result['pep']['AUPR']:.6f}, F1:{result['pep']['F1']:.6f}")
                    print(f"Protein ACC:{result['pro']['accuracy']:.4f}, AUC:{result['pro']['AUC']:.6f}, AUPR:{result['pro']['AUPR']:.6f}, F1:{result['pro']['F1']:.6f}")
                    if result['pep']['F1']+result['pro']['F1'] >= best_score:
                        best_pep_auc = result['pep']['AUC']
                        best_pep_aupr = result['pep']['AUPR']
                        best_pep_f1 = result['pep']['F1']
                        best_pro_auc = result['pro']['AUC']
                        best_pro_aupr = result['pro']['AUPR']
                        best_score = result['pep']['F1'] + result['pro']['F1']
                        print(f'Save model of epoch {e} with {n_fold}-fold cv')
                        checkpoint = {'model': IIDLPepPIRes(), 'model_state_dict': model.state_dict()}
                        torch.save(checkpoint, args.res_saved_path)
                dev_pep_AUC_list.append(best_pep_auc)
                dev_pep_AUPR_list.append(best_pep_aupr)
                dev_pep_F1_list.append(best_pep_f1)
                dev_pro_AUC_list.append(best_pro_auc)
                dev_pro_AUPR_list.append(best_pro_aupr)
                dev_pro_F1_list.append(best_score)
                #保留最后一个模型

            print('pep fold mean auc & aupr & f1', np.mean(dev_pep_AUC_list, axis=0), np.mean(dev_pep_AUPR_list, axis=0), np.mean(dev_pep_F1_list, axis=0))
            print('pro fold mean auc & aupr & f1', np.mean(dev_pro_AUC_list, axis=0), np.mean(dev_pro_AUPR_list, axis=0), np.mean(dev_pro_F1_list, axis=0))
            #print('fold std auc & aupr & f1', np.std(dev_AUC_list, axis=0), np.std(dev_AUPR_list, axis=0), np.std(dev_F1_list, axis=0))
            # mean_auc = np.mean(dev_AUC_list, axis=0)
            # mean_aupr = np.mean(dev_AUPR_list, axis=0)
            # mean_f1 = np.mean(dev_F1_list, axis=0)
            # std_auc = np.std(dev_AUC_list, axis=0)
            # std_aupr = np.std(dev_AUPR_list, axis=0)
            # std_f1 = np.std(dev_F1_list, axis=0)
            # # wandb记录
            # wandb.log({'mean_auc': mean_auc,
            #            'mean_aupr': mean_aupr,
            #            'mean_f1': mean_f1,
            #            'std_auc': std_auc,
            #            'std_aupr': std_aupr,
            #            'std_f1': std_f1})
            wandb.finish()
        else:
            test_dataloader = DataLoader(test_feature, batch_size=args.test_batch_size, shuffle=False,
                                        collate_fn=collate_fn, drop_last=True)
            model_ckpt = load_checkpoint(args.res_load_path, args)
            model_ckpt = model_ckpt.to(device)
            #test_loss, test_acc, test_AUC, test_AUPR, test_f1, test_fpr, test_tpr, test_precision, test_recall = res_test(args, model_ckpt, test_dataloader, device, criterion)
            result = res_test(args, model_ckpt, test_dataloader, device, criterion)
            #print(f"ACC:{test_acc:.4f}, AUC:{test_AUC:.6f}, AUPR:{test_AUPR:.6f}, F1:{test_f1:.6f}")
            print(f"Test ACC:{result['pep']['accuracy']:.4f}, AUC:{result['pep']['AUC']:.6f}, AUPR:{result['pep']['AUPR']:.6f}, F1:{result['pep']['F1']:.6f}")
            print( f"Test ACC:{result['pro']['accuracy']:.4f}, AUC:{result['pro']['AUC']:.6f}, AUPR:{result['pro']['AUPR']:.6f}, F1:{result['pro']['F1']:.6f}")
            #肽
            # 绘制ROC和PR曲线
            plt.figure()
            plt.plot(result['pep']['ROC']['fpr'], result['pep']['ROC']['tpr'], label='ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic of pep_RES')
            plt.legend(loc="lower right")
            plt.show()

            # 绘制Precision-Recall曲线
            plt.figure()
            plt.plot(result['pep']['PR']['recall'], result['pep']['PR']['precision'], label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve of pep_RES')
            plt.legend(loc="upper right")
            plt.show()

            #蛋白质
            # 绘制ROC和PR曲线
            plt.figure()
            plt.plot(result['pro']['ROC']['fpr'], result['pro']['ROC']['tpr'], label='ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic of pro_RES')
            plt.legend(loc="lower right")
            plt.show()

            # 绘制Precision-Recall曲线
            plt.figure()
            plt.plot(result['pro']['PR']['recall'], result['pro']['PR']['precision'], label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve of pro_RES')
            plt.legend(loc="upper right")
            plt.show()

if __name__ == "__main__":
    main()
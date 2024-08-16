import os
import pickle
import cv2
import yaml
import torch
from generate_peptide_features import PepFeature
from generate_protein_features import ProFeature

'''
训练、开发集组成：
训练数据：17514，17514，17514，17514，17512
蛋白质序列   蛋白质结合残基    肽序列    肽结合残基    相互作用标签
验证数据：4378，4378，4378，4378，4380
测试集：
510对
[(蛋白质序列、蛋白结合残基、肽序列、肽结合残基、相互作用标签),(),...]

需要通过肽序列和蛋白质序列得到各自5种特征（这里SCRATCH-1D用不了，实际只有4种特征）
然后得到label，
先分别调用四种特征获取函数，分别写入文件，然后再进行特征读取

x_pep = peptide_feature_dict：{'RTFRQVQSSISDFYD': [18 20 7 18 15 22 15 17 17  8 17  5  7 23  5  0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]}
x_2_pep = peptide_2_feature_dict：{'RTFRQVQSSISDFYD': [6 4 1 6 4 1 4 4 4 1 4 5 1 4 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]}
x_bert_pep = PreTrain：{'RTFRQVQSSISDFYD': array([50,128]的数组), ...}

'''

def convert(input_file, type, task, mode=False):
    '''
    :param input_file: 原始数据，格式为[pro,pro_res,pep,pep_res,label]
    :return: x_pep,x_2_pep,x_diso_pep,x_bert_pep,x_p,x_2_p,x_dense_p,x_bert_p,label
    '''

    #工具加载
    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    method_path = cfg['IIDL-PepPI']
    # scratch = cfg['SCRATCH-1D']
    iupred2a = cfg['IUPred2A']
    ncbiblast = cfg['ncbi-blast']
    nrdb90 = cfg['nrdb90']
    protbert = cfg['ProtBERT']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_out_seq = "example"
    dev_out_seq = "example"
    test_out_proseq = "example/test_protein.fasta"
    test_out_pepseq = "example/test_peptide.fasta"


    features_fold = [] #存放训练和验证数据
    #读取原始数据
    file_list = {}
    for file in os.listdir(input_file):
        aa = file.split(' ')[0]
        f = open(os.path.join(input_file, file), 'rb')
        f = pickle.load(f)
        file_list[file.split(' ')[0]] = f
    #获取序列数据，以及标签
    for item, value in file_list.items():
        if item == 'Training' and type == 'train':
            fold = 5
            for i in range(fold):
                train_data = value[i]
                if mode == True:
                    with open(os.path.join(train_out_seq, f"train_pro_fold_{i}.fasta"), 'w') as f:
                        f.write(f"> Protein_Sequence_fold{i}\n")
                        for seq in train_data:
                            f.write(f"{seq[0]}\n")
                    with open(os.path.join(train_out_seq, f"train_pep_fold_{i}.fasta"), 'w') as f:
                        f.write(f"> Peptide_Sequence_fold{i}\n")
                        for seq in train_data:
                            f.write(f"{seq[2]}\n")
                else:
                    features = []  # 存放特征
                    train_label = []
                    res_len = []
                    pro_res_label = []
                    pep_res_label = []
                    for index, seq in enumerate(train_data):
                        train_label.append(seq[4])
                        pro = str(seq[1])
                        pep = str(seq[3])
                        #分别记录结合残基位置
                        cov_list1 = [int(char) for char in pep]
                        cov_list2 = [int(char) for char in pro]
                        pep_label = [1 if x == 2 else x for x in cov_list1]
                        pro_label = [1 if x == 2 else x for x in cov_list2]
                        res_len.append([len(pep_label), len(pro_label)])
                        if len(pep_label) > 50:
                            pep_label = pep_label[:50]
                        else:
                            pep_label = pep_label + [0] * (50 - len(pep_label))
                        if len(pro_label) > 800:
                            pro_label = pro_label[:800]
                        else:
                            pro_label = pro_label + [0] * (800 - len(pro_label))
                        pep_res_label.append(pep_label)
                        pro_res_label.append(pro_label)
                        # res_gold = pep+pro
                        # res_gold = int(res_gold)
                        # res_label.append(res_gold)

                    # 生成四种特征
                    # 蛋白质序列, 蛋白质列表、序列文件路径、方法路径、五个特征工具路径
                    # ProFeature(prolist, pro_uip, method_path, protbert, iupred2a, ncbiblast, nrdb90, device):
                    train_out_pepseq = os.path.join(train_out_seq, f"train_pep_fold_{i}.fasta")
                    train_out_proseq = os.path.join(train_out_seq, f"train_pro_fold_{i}.fasta")
                    pep_seq_list, pro_seq_list = [], []
                    for i in open(train_out_pepseq):
                        if i[0] != '>':
                            pep_seq_list.append(i.strip())
                    for i in open(train_out_proseq):
                        if i[0] != '>':
                            pro_seq_list.append(i.strip())
                    # pep_seq_list = pep_seq_list[:100]
                    # pro_seq_list = pro_seq_list[:100]
                    x_p, x_2_p = ProFeature(pro_seq_list, test_out_proseq, method_path, protbert, iupred2a, ncbiblast, nrdb90, device)
                    x_pep, x_2_pep = PepFeature(pep_seq_list, test_out_pepseq, method_path,protbert, iupred2a, ncbiblast, nrdb90, device)
                    sample_number = len(train_data)
                    #sample_number = 100
                    # 根据序列索引，从对应特征中提取对应特征
                    if task == "seq":
                        for idx in range(sample_number):
                            feature = {'x_pep': x_pep[pep_seq_list[idx]],  # [,,,]
                                       'x_2_pep': x_2_pep[pep_seq_list[idx]],
                                       # 'x_dense_pep': x_dense_pep[pep_seq_list[idx]],
                                       # 'x_bert_pep': x_bert_pep[pep_seq_list[idx]],
                                       'x_p': x_p[pro_seq_list[idx]],
                                       'x_2_p': x_2_p[pro_seq_list[idx]],
                                       # 'x_dense_p': x_dense_p[pro_seq_list[idx]],
                                       # 'x_bert_p': x_bert_p[pro_seq_list[idx]],
                                       'label': [train_label[idx]],
                                       'pep_res_label': pep_res_label[idx],
                                       'pro_res_label': pro_res_label[idx],
                                       'res_len': res_len[idx],
                                       # 'res_label': [res_label[idx]]
                                       }
                            features.append(feature)
                        features_fold.append(features)
                    else:
                        for idx in range(sample_number):
                            if train_label[idx] == 1:
                                feature = {'x_pep': x_pep[pep_seq_list[idx]],  # [,,,]
                                           'x_2_pep': x_2_pep[pep_seq_list[idx]],
                                           # 'x_dense_pep': x_dense_pep[pep_seq_list[idx]],
                                           # 'x_bert_pep': x_bert_pep[pep_seq_list[idx]],
                                           'x_p': x_p[pro_seq_list[idx]],
                                           'x_2_p': x_2_p[pro_seq_list[idx]],
                                           # 'x_dense_p': x_dense_p[pro_seq_list[idx]],
                                           # 'x_bert_p': x_bert_p[pro_seq_list[idx]],
                                           'label': [train_label[idx]],
                                           'pep_res_label': pep_res_label[idx],
                                           'pro_res_label': pro_res_label[idx],
                                           'res_len': res_len[idx],
                                           # 'res_label': [res_label[idx]]
                                           }
                                features.append(feature)
                            else:
                                continue
                        features_fold.append(features)
            return features_fold
        elif item == 'Validation' and type == 'dev':
            #5折验证集
            fold = 5
            for i in range(fold):
                dev_data = value[i]
                if mode == True:
                    with open(os.path.join(dev_out_seq, f"dev_pro_fold_{i}.fasta"), 'w') as f:
                        f.write(f"> Protein_Sequence_fold{i}\n")
                        for seq in dev_data:
                            f.write(f"{seq[0]}\n")
                    with open(os.path.join(dev_out_seq, f"dev_pep_fold_{i}.fasta"), 'w') as f:
                        f.write(f"> Peptide_Sequence_fold{i}\n")
                        for seq in dev_data:
                            f.write(f"{seq[2]}\n")
                else:
                    features = []  # 存放特征
                    dev_label = []
                    #res_label = []
                    pro_res_label = []
                    pep_res_label = []
                    res_len = []
                    for index, seq in enumerate(dev_data):
                        dev_label.append(seq[4])
                        pro = str(seq[1])
                        pep = str(seq[3])
                        # 分别记录结合残基位置
                        cov_list1 = [int(char) for char in pep]
                        cov_list2 = [int(char) for char in pro]
                        pep_label = [1 if x == 2 else x for x in cov_list1]
                        pro_label = [1 if x == 2 else x for x in cov_list2]
                        res_len.append([len(pep_label), len(pro_label)])
                        if len(pep_label) > 50:
                            pep_label = pep_label[:50]
                        else:
                            pep_label = pep_label + [0] * (50 - len(pep_label))
                        if len(pro_label) > 800:
                            pro_label = pro_label[:800]
                        else:
                            pro_label = pro_label + [0] * (800 - len(pro_label))
                        pep_res_label.append(pep_label)
                        pro_res_label.append(pro_label)
                    # for index, seq in enumerate(dev_data):
                    #     dev_label.append(seq[4])
                    #     pro = str(seq[1])
                    #     pep = str(seq[3])
                    #     res_gold = pep + pro
                    #     res_gold = int(res_gold)
                    #     res_label.append(res_gold)

                    # 生成四种特征
                    # 蛋白质序列, 蛋白质列表、序列文件路径、方法路径、五个特征工具路径
                    # ProFeature(prolist, pro_uip, method_path, protbert, iupred2a, ncbiblast, nrdb90, device):
                    dev_out_pepseq = os.path.join(dev_out_seq, f"dev_pep_fold_{i}.fasta")
                    dev_out_proseq = os.path.join(dev_out_seq, f"dev_pro_fold_{i}.fasta")
                    pep_seq_list, pro_seq_list = [], []
                    for i in open(dev_out_pepseq):
                        if i[0] != '>':
                            pep_seq_list.append(i.strip())
                    for i in open(dev_out_proseq):
                        if i[0] != '>':
                            pro_seq_list.append(i.strip())
                    # pep_seq_list = pep_seq_list[:100]
                    # pro_seq_list = pro_seq_list[:100]
                    x_p, x_2_p = ProFeature(pro_seq_list, test_out_proseq, method_path, protbert, iupred2a,
                                                                 ncbiblast, nrdb90, device)
                    x_pep, x_2_pep = PepFeature(pep_seq_list, test_out_pepseq, method_path, protbert,
                                                                         iupred2a,ncbiblast, nrdb90, device)
                    #一条蛋白序列对应一条肽段序列，然后一个feature存放一对序列的信息

                    sample_number = len(dev_data)
                    #sample_number = 100
                    # 根据序列索引，从对应特征中提取对应特征
                    if task == "seq":
                        for idx in range(sample_number):
                            feature = {'x_pep': x_pep[pep_seq_list[idx]],  # [,,,]
                                       'x_2_pep': x_2_pep[pep_seq_list[idx]],
                                       # 'x_dense_pep': x_dense_pep[pep_seq_list[idx]],
                                       # 'x_bert_pep': x_bert_pep[pep_seq_list[idx]],
                                       'x_p': x_p[pro_seq_list[idx]],
                                       'x_2_p': x_2_p[pro_seq_list[idx]],
                                       # 'x_dense_p': x_dense_p[pro_seq_list[idx]],
                                       # 'x_bert_p': x_bert_p[pro_seq_list[idx]],
                                       'label': [dev_label[idx]],
                                       'pep_res_label': pep_res_label[idx],
                                       'pro_res_label': pro_res_label[idx],
                                       'res_len': res_len[idx],
                                       #'res_label': [res_label[idx]],
                                       }
                            features.append(feature)
                    else:
                        for idx in range(sample_number):
                            if dev_label[idx] == 1:
                                feature = {'x_pep': x_pep[pep_seq_list[idx]],  # [,,,]
                                           'x_2_pep': x_2_pep[pep_seq_list[idx]],
                                           # 'x_dense_pep': x_dense_pep[pep_seq_list[idx]],
                                           # 'x_bert_pep': x_bert_pep[pep_seq_list[idx]],
                                           'x_p': x_p[pro_seq_list[idx]],
                                           'x_2_p': x_2_p[pro_seq_list[idx]],
                                           # 'x_dense_p': x_dense_p[pro_seq_list[idx]],
                                           # 'x_bert_p': x_bert_p[pro_seq_list[idx]],
                                           'label': [dev_label[idx]],
                                           'pep_res_label': pep_res_label[idx],
                                           'pro_res_label': pro_res_label[idx],
                                           'res_len': res_len[idx],
                                           #'res_label': [res_label[idx]],
                                           }
                                features.append(feature)
                            else:
                                continue
                    features_fold.append(features)
            return features_fold
        elif item == 'Test' and type == 'test':
            test_data = value #[(pro, pro_res, pep, pep_res, label), (pro, pro_res, pep, pep_res, label), ...],510对
            if mode == True:
                with open(test_out_proseq, 'w') as f:
                    f.write(f"> Protein_Sequence_510\n")
                    for i, seq in enumerate(test_data):
                        f.write(f"{seq[0]}\n")
                with open(test_out_pepseq, 'w') as f:
                    f.write(f"> Peptide_Sequence_510\n")
                    for i, seq in enumerate(test_data):
                        f.write(f"{seq[2]}\n")
            else:
                features = []  # 存放特征
                #获取测试集标签
                test_label = []
                res_label = []
                pro_res_label = []
                pep_res_label = []
                res_len = []
                for index, seq in enumerate(test_data):
                    test_label.append(seq[4])
                    pro = str(seq[1])
                    pep = str(seq[3])
                    # 分别记录结合残基位置
                    cov_list1 = [int(char) for char in pep]
                    cov_list2 = [int(char) for char in pro]
                    pep_label = [1 if x == 2 else x for x in cov_list1]
                    pro_label = [1 if x == 2 else x for x in cov_list2]
                    res_len.append([len(pep_label), len(pro_label)])
                    if len(pep_label) > 50:
                        pep_label = pep_label[:50]
                    else:
                        pep_label = pep_label + [0] * (50 - len(pep_label))
                    if len(pro_label) > 800:
                        pro_label = pro_label[:800]
                    else:
                        pro_label = pro_label + [0] * (800 - len(pro_label))
                    pep_res_label.append(pep_label)
                    pro_res_label.append(pro_label)
                # for i, seq in enumerate(test_data):
                #     test_label.append(seq[4])
                #     pro = str(seq[1])
                #     pep = str(seq[3])
                #     res_gold = pep + pro
                #     res_gold = int(res_gold)
                #     res_label.append(res_gold)

                #生成四种特征
                #蛋白质序列, 蛋白质列表、序列文件路径、方法路径、五个特征工具路径
                #ProFeature(prolist, pro_uip, method_path, protbert, iupred2a, ncbiblast, nrdb90, device):

                pep_seq_list, pro_seq_list = [], []
                name_list = []

                for i in open(test_out_pepseq):
                    if i[0] != '>':
                        pep_seq_list.append(i.strip())
                for i in open(test_out_proseq):
                    if i[0] != '>':
                        pro_seq_list.append(i.strip())
                # pep_seq_list = pep_seq_list[:100]
                # pro_seq_list = pro_seq_list[:100]
                #会有重复蛋白质序列和肽序列，对于每一对，需要根据初始蛋白质序列和肽序列的索引进行特征提取
                x_p, x_2_p = ProFeature(pro_seq_list, test_out_proseq, method_path, protbert, iupred2a,
                                                             ncbiblast, nrdb90, device)
                x_pep, x_2_pep = PepFeature(pep_seq_list, test_out_pepseq, method_path, protbert, iupred2a,
                                                             ncbiblast, nrdb90, device)
                sample_number = len(test_data)
                #sample_number = 100
                #根据序列索引，从对应特征中提取对应特征
                if task == "seq":
                    for idx in range(sample_number):
                        feature = {'x_pep': x_pep[pep_seq_list[idx]], #[,,,]
                                   'x_2_pep': x_2_pep[pep_seq_list[idx]],
                                   # 'x_dense_pep': x_dense_pep[pep_seq_list[idx]],
                                   # 'x_bert_pep': x_bert_pep[pep_seq_list[idx]],
                                   'x_p': x_p[pro_seq_list[idx]],
                                   'x_2_p': x_2_p[pro_seq_list[idx]],
                                   # 'x_dense_p': x_dense_p[pro_seq_list[idx]],
                                   # 'x_bert_p': x_bert_p[pro_seq_list[idx]],
                                   'label': [test_label[idx]],
                                   'pep_res_label': pep_res_label[idx],
                                   'pro_res_label': pro_res_label[idx],
                                   'res_len': res_len[idx],
                                   #'res_label': [res_label[idx]],
                                   }
                        features.append(feature)
                else:
                    for idx in range(sample_number):
                        if test_label[idx] == 1:
                            feature = {'x_pep': x_pep[pep_seq_list[idx]], #[,,,]
                                       'x_2_pep': x_2_pep[pep_seq_list[idx]],
                                       # 'x_dense_pep': x_dense_pep[pep_seq_list[idx]],
                                       # 'x_bert_pep': x_bert_pep[pep_seq_list[idx]],
                                       'x_p': x_p[pro_seq_list[idx]],
                                       'x_2_p': x_2_p[pro_seq_list[idx]],
                                       # 'x_dense_p': x_dense_p[pro_seq_list[idx]],
                                       # 'x_bert_p': x_bert_p[pro_seq_list[idx]],
                                       'label': [test_label[idx]],
                                       'pep_res_label': pep_res_label[idx],
                                       'pro_res_label': pro_res_label[idx],
                                       'res_len': res_len[idx],
                                       #'res_label': [res_label[idx]],
                                       }
                            features.append(feature)
                        else:
                            continue
                return features
    return 0

if __name__ == '__main__':
    path = "F:/papercode/IIDL-PepPI-main/datasets/5-fold Randomly Split Datasets"
    process_data = convert(path, 'dev','seq')



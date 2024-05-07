import os, random, math
import numpy as np
from argparse import ArgumentParser
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from utils.parsing import add_train_args, modify_train_args
from utils.util import makedirs,get_loss_func,get_metric_func
from utils.metrics_graph import metrics_graph
from utils.pack import pack,pack_combo
import pickle
import torch
from data_set_split.data_split import data_split
from data_process.smiles2adj import single_process, combo_process_noid,combo_process
import pandas as pd
from model.DiseaseModel import DiseaseModel_mlp

# from data_process.pro.amino import extract_input_data,batch2tensor,protein_process



def get_dataset(args):
    single_path = args.echi_path
    val_path = args.echi_combo_val
    args.num_tasks = 1

    # 读取单药数据
    with open(single_path, 'rb') as file_s:
        single_dataset = pickle.load(file_s)
    args.train_data_size = len(single_dataset)
    # 读取训练数据
    with open(args.echi_combo_train, 'rb') as file_train:
        train_dataset = pickle.load(file_train)
    # 读取验证集数据
    with open(val_path, 'rb') as file_val:
        val_dataset = pickle.load(file_val)
    return single_dataset, train_dataset, val_dataset

def train(single_dataset, train_dataset, model, optimizer, loss_func, args):
    model.train()
    random.shuffle(single_dataset)
    random.shuffle(train_dataset)
    # 所有单药的数据处理
    adjs, atom_features, labels = [], [], []
    for single_data in single_dataset:
        adj, atom_feature, label = single_data
        adjs.append(adj)
        atom_features.append(atom_feature)
        labels.append(label)  # labels为tensor 可以直接拆分 3113维
    atom_features, adjs, labels = pack(atom_features, adjs, labels)  # labels（3113，）
    labels = labels.cpu().numpy().tolist()  # list [0,0,0.....]
    adjs_single_new, features_single = [], []
    for i in range(adjs.size(0)):
        adjs_single_new.append(adjs[i, :, :])
    for i in range(atom_features.size(0)):
        features_single.append(atom_features[i, :, :])

    # 分批次训练
    for i in trange(0, len(adjs_single_new), args.batch_size):
        random.shuffle(train_dataset)
        # 所有药物组合的数据处理
        adjs_d1, features_d1, adjs_d2, features_d2, labels_combo = [], [], [], [], []  # 用列表存放所有元素各自的矩阵和特征
        for train_data in train_dataset:
            adj_d1, atom_feature_d1, adj_d2, atom_feature_d2, label_combo = train_data
            adjs_d1.append(adj_d1)
            features_d1.append(atom_feature_d1)
            adjs_d2.append(adj_d2)
            features_d2.append(atom_feature_d2)
            labels_combo.append(label_combo)
        adjs_d1, features_d1, adjs_d2, features_d2, labels_combo = pack_combo(adjs_d1, features_d1, adjs_d2,
                                                                              features_d2, labels_combo)
        labels_combo = labels_combo.cpu().numpy().tolist()
        adjs1_combo_new, features1_combo, adjs2_combo_new, features2_combo = [], [], [], []
        for i in range(adjs_d1.size(0)):  # 转化为拼接的矩阵和特征，每个图的维度都是一样的，存放在列表中
            adjs1_combo_new.append(adjs_d1[i, :, :])
        for i in range(adjs_d2.size(0)):
            adjs2_combo_new.append(adjs_d2[i, :, :])
        for i in range(features_d1.size(0)):
            features1_combo.append(features_d1[i, :, :])
        for i in range(features_d2.size(0)):
            features2_combo.append(features_d2[i, :, :])

        # # 清空模型的梯度信息的操作,保证每个batch不受上一个batch的参数影响
        model.zero_grad()
        # 单药批处理
        adj_s = adjs_single_new[i:i + args.batch_size]
        adj_s = torch.stack(adj_s)
        feature_s = features_single[i:i + args.batch_size]
        feature_s = torch.stack(feature_s)
        label_s = labels[i:i + args.batch_size]  # list [0,0,0,,,,]

        # 药物组合的批处理
        adj1_combo_batch = adjs1_combo_new[:args.batch_size]
        adj1_combo_batch = torch.stack(adj1_combo_batch)
        adj2_combo_batch = adjs2_combo_new[:args.batch_size]
        adj2_combo_batch = torch.stack(adj2_combo_batch)
        features1_combo_batch = features1_combo[:args.batch_size]
        features1_combo_batch = torch.stack(features1_combo_batch)
        features2_combo_batch = features2_combo[:args.batch_size]
        features2_combo_batch = torch.stack(features2_combo_batch)
        labels_combo_batch = labels_combo[:args.batch_size]

        if len(adj_s) < args.batch_size:
            continue

        # single train
        # mask = torch.Tensor([[tb] for tb in label_s]).cuda()
        label_s = torch.Tensor([[tb] for tb in label_s]).cuda()
        label_s = label_s.squeeze()
        preds = model(args.batch_size, feature_s, adj_s)
        preds = preds.squeeze()
        echi_single_loss = loss_func(preds, label_s)
        echi_single_loss = torch.mean(echi_single_loss)
        label_s = mask = None

        # combo train
        # mask = torch.Tensor([[tb] for tb in labels_combo_batch]).cuda()
        label_combo = torch.Tensor([[tb] for tb in labels_combo_batch]).cuda()
        preds = model.combo_forward(args.batch_size, features1_combo_batch, adj1_combo_batch, features2_combo_batch,
                                    adj2_combo_batch)
        echi_combo_loss = loss_func(preds, label_combo)
        echi_combo_loss = torch.mean(echi_combo_loss)
        # echi_combo_loss = (echi_combo_loss * mask).sum() / mask.sum()
        label_combo = mask = None

        loss = args.single_lambda * echi_single_loss + args.combo_lambda * echi_combo_loss

        loss.backward()
        optimizer.step()
        # scheduler.step()
        # print("epoch:  "+str(i),"{: .4f}".format(loss))
        # 如果在每个批次中不涉及loss，但是对没批数据都要训练到最好，

from sklearn.utils import shuffle
def combo_evaluate(model, dataset, args):
    model.eval()
    all_pred, all_target = [], []
    dataset = shuffle(dataset)
    # 所有药物组合的数据处理
    adjs_d1, features_d1, adjs_d2, features_d2, labels_combo = [], [], [], [], []  # 用列表存放所有元素各自的矩阵和特征
    for data in dataset:
        adj_d1, atom_feature_d1, adj_d2, atom_feature_d2, label_combo = data
        adjs_d1.append(adj_d1)
        features_d1.append(atom_feature_d1)
        adjs_d2.append(adj_d2)
        features_d2.append(atom_feature_d2)
        labels_combo.append(label_combo)
    adjs_d1, features_d1, adjs_d2, features_d2, labels_combo = pack_combo(adjs_d1, features_d1, adjs_d2, features_d2,
                                                                          labels_combo)
    labels_combo = labels_combo.cpu().numpy().tolist()
    adjs1_combo_new, features1_combo, adjs2_combo_new, features2_combo = [], [], [], []
    for i in range(adjs_d1.size(0)):  # 转化为拼接的矩阵和特征，每个图的维度都是一样的，存放在列表中
        adjs1_combo_new.append(adjs_d1[i, :, :])
    for i in range(adjs_d2.size(0)):
        adjs2_combo_new.append(adjs_d2[i, :, :])
    for i in range(features_d1.size(0)):
        features1_combo.append(features_d1[i, :, :])
    for i in range(features_d2.size(0)):
        features2_combo.append(features_d2[i, :, :])

    for i in trange(0, len(dataset), args.batch_size):
        # 药物组合的批处理
        adj1_combo_batch = adjs1_combo_new[:args.batch_size]
        adj1_combo_batch = torch.stack(adj1_combo_batch)
        adj2_combo_batch = adjs2_combo_new[:args.batch_size]
        adj2_combo_batch = torch.stack(adj2_combo_batch)
        features1_combo_batch = features1_combo[:args.batch_size]
        features1_combo_batch = torch.stack(features1_combo_batch)
        features2_combo_batch = features2_combo[:args.batch_size]
        features2_combo_batch = torch.stack(features2_combo_batch)
        labels_combo_batch = labels_combo[:args.batch_size]

        label_combo = torch.Tensor([[tb] for tb in labels_combo_batch]).cuda()

        preds = model.combo_forward(len(dataset), features1_combo_batch, adj1_combo_batch, features2_combo_batch,
                                    adj2_combo_batch)

        all_pred.extend(preds.tolist())
        # all_target.extend(label_combo)
        all_preds = [item for sublist in all_pred for item in sublist]
        # all_targets = [item for sublist in all_target for item in sublist]
        all_targets = label_combo.squeeze().tolist()

    auc, aupr, f1_score, accuracy, recall, precision = metrics_graph(all_targets, all_preds)
    return auc, aupr, f1_score, accuracy, recall, precision

def run_training(args, save_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    single_dataset, train_dataset, val_dataset = get_dataset(args)
    model = DiseaseModel_mlp(args).cuda()
    loss_func = get_loss_func(args)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = WarmupLinearSchedule(optimizer,
    #         warmup_steps=args.train_data_size / args.batch_size * 2,
    #         t_total=args.train_data_size / args.batch_size * args.epochs
    # )
    args.metric_func = get_metric_func(metric=args.metric)

    best_auc, best_aupr, best_f1, best_acc, best_recall,best_precision = 0, 0, 0, 0, 0, 0
    for epoch in range(30):
        print(f'Epoch {epoch}')
        train(single_dataset, train_dataset, model, optimizer,loss_func, args)
        # auc, aupr, f1_score, accuracy, recall, specificity, precision = combo_evaluate(model, val_dataset, args)
        auc, aupr, f1_score, accuracy, recall, precision = combo_evaluate(model, val_dataset, args)
        avg_val_auc = np.nanmean(auc)
        avg_val_aupr = np.nanmean(aupr)
        avg_val_f1 = np.nanmean(f1_score)
        avg_val_acc = np.nanmean(accuracy)
        avg_val_recall = np.nanmean(recall)
        avg_val_precision = np.nanmean(precision)
        print(f'epoch{epoch} - Combo Validation {args.metric} ={avg_val_auc:.4f}')
        print(f'epoch{epoch} - Combo Validation aupr = {avg_val_aupr:.4f}')
        print(f'epoch{epoch} - Combo Validation f1 = {avg_val_f1:.4f}')
        print(f'epoch{epoch} - Combo Validation acc = {avg_val_acc:.4f}')
        print(f'epoch{epoch} - Combo Validation recall = {avg_val_recall:.4f}')
        print(f'epoch{epoch} - Combo Validation precision = {avg_val_precision:.4f}')
        if avg_val_auc > best_auc :
            best_auc = avg_val_auc
            best_aupr = avg_val_aupr
            best_f1 = avg_val_f1
            best_acc = avg_val_acc
            best_recall = avg_val_recall
            best_precision = avg_val_precision
        print(f'epoch{epoch} - Combo Validation best{args.metric} ={best_auc:.4f}')
        print(f'epoch{epoch} - Combo Validation best aupr = {best_aupr:.4f}')
        print(f'epoch{epoch} - Combo Validation best f1 = {best_f1:.4f}')
        print(f'epoch{epoch} - Combo Validation best acc = {best_acc:.4f}')
        print(f'epoch{epoch} - Combo Validation best recall = {best_recall:.4f}')
        print(f'epoch{epoch} - Combo Validation precision = {best_precision:.4f}')

    return best_auc, best_aupr, best_f1, best_acc, best_recall,best_precision

def main(args):


    all_test_scores = [0] * args.num_folds
    all_test_ap = [0] * args.num_folds
    all_test_aupr = [0] * args.num_folds
    all_test_f1 = [0] * args.num_folds
    all_test_acc = [0] * args.num_folds
    all_test_recall = [0] * args.num_folds
    all_test_spec = [0] * args.num_folds
    for i in range(0, args.num_folds):
        fold_dir = os.path.join(args.save_dir, f'fold_{i}')
        makedirs(fold_dir)
        args.seed = i
        all_test_scores[i], all_test_aupr[i], all_test_f1[i], all_test_acc[i], all_test_recall[i], all_test_ap[i] = run_training(args,fold_dir)

    all_test_scores = np.stack(all_test_scores, axis=0)
    all_test_aupr = np.stack(all_test_aupr, axis=0)
    all_test_f1 = np.stack(all_test_f1, axis=0)
    all_test_acc = np.stack(all_test_acc, axis=0)
    all_test_recall = np.stack(all_test_recall, axis=0)
    all_test_spec = np.stack(all_test_spec, axis=0)
    all_test_ap = np.stack(all_test_ap, axis=0)
    mean, std = np.mean(all_test_scores, axis=0), np.std(all_test_scores, axis=0)
    mean_aupr, std_aupr = np.mean(all_test_aupr, axis=0), np.std(all_test_aupr, axis=0)
    mean_f1, std_f1 = np.mean(all_test_f1, axis=0), np.std(all_test_f1, axis=0)
    mean_acc, std_acc = np.mean(all_test_acc, axis=0), np.std(all_test_acc, axis=0)
    mean_recall, std_recall = np.mean(all_test_recall, axis=0), np.std(all_test_recall, axis=0)
    mean_spec, std_spec = np.mean(all_test_spec, axis=0), np.std(all_test_spec, axis=0)
    mean_ap, std_ap = np.mean(all_test_ap, axis=0), np.std(all_test_ap, axis=0)
    print(f'auc {args.num_folds} fold average: {mean} +/- {std}')
    print(f'aupr {args.num_folds} fold average: {mean_aupr} +/- {std_aupr}')
    print(f'f1 {args.num_folds} fold average: {mean_f1} +/- {std_f1}')
    print(f'acc {args.num_folds} fold average: {mean_acc} +/- {std_acc}')
    print(f'recall {args.num_folds} fold average: {mean_recall} +/- {std_recall}')
    print(f'spec{args.num_folds} fold average: {mean_spec} +/- {std_spec}')
    print(f'precision {args.num_folds} fold average: {mean_ap} +/- {std_ap}')


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
        steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        # print(float(step) / float(max(1, self.warmup_steps)))
        # print(max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps))))
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))

if __name__ == "__main__":
    combo_path = '/root/autodl-tmp/jly/echi/data/ae/ae_synergy.csv'  # 正负比1：4
    random_seed = 42
    folder_name = f'/root/autodl-tmp/jly/echi/data/ae/'
    testdataset, traindataset = data_split(combo_path, random_seed=42,folder_name=folder_name)
    test_path = testdataset
    train_path = traindataset
    single_path = "/root/autodl-tmp/jly/echi/data/ae/ae_single.csv"
    path = f"/root/autodl-tmp/jly/echi/data_process/ae/"
    test_process_nn, train_process_nn = combo_process_noid(test_path, train_path,path=path)
    single_process_nn = single_process(single_path,path=path)


    parser = ArgumentParser()
    parser.add_argument('--echi_path', default=single_process_nn)
    parser.add_argument('--echi_combo_train', default=train_process_nn)
    parser.add_argument('--echi_combo_val', default=test_process_nn)
    parser.add_argument('--single_lambda', type=float, default=0.1)
    parser.add_argument('--combo_lambda', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--cells_size', type=int, default=64)
    parser.add_argument('--use_GMP', type=bool, default=True)
    # parser.add_argument('--n_amino', type=int, default=len(amino_dict))


    add_train_args(parser)
    args = parser.parse_args()
    args.dataset_type = 'classification'
    args.num_folds = 5

    modify_train_args(args)
    print(args)

    main(args)


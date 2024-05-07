import numpy as np
from rdkit import Chem
import torch
import pickle
import sys
import csv
from data_set_split.data_split import data_split
import pandas as pd
sys.path.append('..')
num_atom_feat = 34

def combo_process_noid(test_path,train_path,path = None):
    # test_path = testdataset
    smiles1_list = []
    smiles2_list = []
    target_list = []
    adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], []

    with open(test_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            smiles1,smiles2, target = row[0], row[1], row[2]
            smiles1_list.append(smiles1)
            smiles2_list.append(smiles2)
            target_list.append(target)

            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(target, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
            except:
                continue

        print(f"test--Smiles1 元素总数: {len(smiles1_list)}")
        print(f"test--Smiles1 元素总数: {len(smiles2_list)}")
        print(f"test--Label 元素总数: {len(target_list)}")

    dataset = list(zip(adjacencies1, features1, adjacencies2, features2, labels))

    # test_process_nn = "/home/jly/echico/data_process/ce/test.pickle" # ce正负1：2
    test_process_nn =path + "test.pickle" # ce正负1：2
    print(test_process_nn)
    with open(test_process_nn, "wb") as f:
        pickle.dump(dataset, f)

    # 将traindata处理为adj，fea和label

    smiles1_list = []
    smiles2_list = []

    target_list = []
    adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], []

    with open(train_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            smiles1, smiles2, target = row[0], row[1], row[2]
            smiles1_list.append(smiles1)
            smiles2_list.append(smiles2)
            target_list.append(target)

            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(target, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
            except:
                continue

        print(f"train--Smiles1 元素总数: {len(smiles1_list)}")
        print(f"train--Smiles1 元素总数: {len(smiles2_list)}")
        print(f"train--Label 元素总数: {len(target_list)}")

    dataset = list(zip(adjacencies1, features1, adjacencies2, features2, labels))

    # train_process_nn = "/home/jly/echico/data_process/ce/trainnn.pickle"  # ce 1:2
    train_process_nn = path + "trainnn.pickle"


    with open(train_process_nn, "wb") as f:
        pickle.dump(dataset, f)

    return test_process_nn,train_process_nn

def combo_process(test_path,train_path,path = None):
    ids1,adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], [],[]
    smiles1_list_train = []
    smiles2_list_train = []
    target_list_train = []
    id_list_train = []
    with open(train_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            id,smiles1,smiles2, label = row[0], row[1], row[2],row[3]
            smiles1_list_train.append(smiles1)
            smiles2_list_train.append(smiles2)
            target_list_train.append(label)
            id_list_train.append(id)
            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(label, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
                ids1.append(id)
            except:
                continue

        # 读取每一行的五个元素
        # id = row.loc['CID']  # 假设列名称为 column1, column2, ..., column5
        # smiles1 = row.loc['smiles1']
        # smiles2 = row.loc['smiles2']
        # label = row.loc['label']
        # id_list_train.append(id)
        # smiles1_list_train.append(smiles1)
        # smiles2_list_train.append(smiles2)
        # target_list_train.append(label)
    trian_dataset = list(zip(ids1,adjacencies1, features1, adjacencies2, features2, labels))
    print("train数量为：", len(smiles1_list_train))
    print("train数量为：", len(trian_dataset))
    train_process_nn = path + "trainnn.pickle"
    with open(train_process_nn, "wb") as f:
        pickle.dump(trian_dataset, f)


    ids1,adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], [],[]
    smiles1_list_val = []
    smiles2_list_val = []
    target_list_val = []
    id_list_val = []
    with open(test_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            id,smiles1,smiles2, label = row[0], row[1], row[2],row[3]
            smiles1_list_val.append(smiles1)
            smiles2_list_val.append(smiles2)
            target_list_val.append(label)
            id_list_val.append(id)
            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(label, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
                ids1.append(id)
            except:
                continue


    val_dataset = list(zip(ids1,adjacencies1, features1, adjacencies2, features2, labels))
    print("val数量为：",len(smiles1_list_val))
    print("val数量为：",len(val_dataset))

    val_process_nn = path + "valnn.pickle"
    with open(val_process_nn, "wb") as f:
        pickle.dump(val_dataset, f)
    return val_process_nn,train_process_nn

def single_process(single_path,path= None):

    smiles_list = []
    target_list = []
    adjacencies1,features1,labels = [],[],[]

    with open(single_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            smiles, target = row[0], row[1]
            smiles_list.append(smiles)
            target_list.append(target)

            try:
                atom_feature1, adj1 = mol_features(smiles)
                label = np.array(target, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)

                labels.append(label)
            except:
                continue

        print(f"单药Smiles1 元素总数: {len(smiles_list)}")

        print(f"单药Label 元素总数: {len(target_list)}")

    dataset = list(zip(adjacencies1,features1,labels))

    # single_process_nn = "/home/jly/echi2/data_process/ce/single.pickle"  # ce 正负1：2
    single_process_nn = path + "single.pickle"

    with open(single_process_nn, "wb") as f:
        pickle.dump(dataset, f)
    return single_process_nn

def process(test_path,train_path,val_path,path = None):
    # test_path = testdataset
    smiles1_list = []
    smiles2_list = []
    target_list = []
    adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], []

    with open(test_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            smiles1,smiles2, target = row[0], row[1], row[2]
            smiles1_list.append(smiles1)
            smiles2_list.append(smiles2)
            target_list.append(target)

            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(target, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
            except:
                continue

        print(f"test--Smiles1 元素总数: {len(smiles1_list)}")
        print(f"test--Smiles1 元素总数: {len(smiles2_list)}")
        print(f"test--Label 元素总数: {len(target_list)}")
    dataset = list(zip(adjacencies1, features1, adjacencies2, features2, labels))
    test_process_nn =path + "test.pickle" # ce正负1：2
    with open(test_process_nn, "wb") as f:
        pickle.dump(dataset, f)


    smiles1_list = []
    smiles2_list = []
    target_list = []
    adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], []

    with open(train_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            smiles1, smiles2, target = row[0], row[1], row[2]
            smiles1_list.append(smiles1)
            smiles2_list.append(smiles2)
            target_list.append(target)

            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(target, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
            except:
                continue

        print(f"train--Smiles1 元素总数: {len(smiles1_list)}")
        print(f"train--Smiles1 元素总数: {len(smiles2_list)}")
        print(f"train--Label 元素总数: {len(target_list)}")

    dataset = list(zip(adjacencies1, features1, adjacencies2, features2, labels))
    train_process_nn = path + "train.pickle"
    with open(train_process_nn, "wb") as f:
        pickle.dump(dataset, f)

    smiles1_list = []
    smiles2_list = []
    target_list = []
    adjacencies1, features1, adjacencies2, features2, labels = [], [], [], [], []

    with open(val_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            smiles1, smiles2, target = row[0], row[1], row[2]
            smiles1_list.append(smiles1)
            smiles2_list.append(smiles2)
            target_list.append(target)

            try:
                atom_feature1, adj1 = mol_features(smiles1)
                atom_feature2, adj2 = mol_features(smiles2)

                label = np.array(target, dtype=np.float32)
                atom_feature1 = torch.FloatTensor(atom_feature1)
                adj1 = torch.FloatTensor(adj1)

                atom_feature2 = torch.FloatTensor(atom_feature2)
                adj2 = torch.FloatTensor(adj2)
                label = torch.LongTensor(label)

                features1.append(atom_feature1)
                adjacencies1.append(adj1)
                features2.append(atom_feature2)
                adjacencies2.append(adj2)

                labels.append(label)
            except:
                continue

        print(f"val--Smiles1 元素总数: {len(smiles1_list)}")
        print(f"val--Smiles1 元素总数: {len(smiles2_list)}")
        print(f"val--Label 元素总数: {len(target_list)}")

    dataset = list(zip(adjacencies1, features1, adjacencies2, features2, labels))
    val_process_nn = path + "val.pickle"  # ce正负1：2

    with open(val_process_nn, "wb") as f:
        pickle.dump(dataset, f)
    return test_process_nn,train_process_nn,val_process_nn



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,explicit_H=False,use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    # num_atom_feat = mol.GetNumAtoms()
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def get_pretrain_smi(smi2id, smi, max_len=100):
    smi_idx = [smi2id.get(i, len(smi2id)+1) for i in smi]
    return smi_idx

def drug_feature_extract(drug_data):
    drug_data = pd.DataFrame(drug_data).T
    drug_feat = [[] for _ in range(len(drug_data))]
    for i in range(len(drug_feat)):
        feat_mat, adj_list = drug_data.iloc[i]
        drug_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return drug_feat
def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]
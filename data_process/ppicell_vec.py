import torch
import torch.nn as nn
from argparse import ArgumentParser
import torch.nn.functional as F
import pickle
import os

device = torch.device("cuda:0")
emb_dim = 64   # 疾病特征维度
aggregation_function =nn.Linear(emb_dim*3, emb_dim).to(device)  # 用于拼接 有多少阶蛋白，emb_dim*几

def main(echi_batch):
    # 要在这里设计一个可以根据数据集选择的对应参数的，比如file_path和 num_1hop_target
    file_path = '/home/jly/echi_final/data/pro/protein_fea_ce.pkl'  # ce
    path = '/home/jly/echi2/data/pro/'  # ae  蛋白质的特征向量
    path = '/root/autodl-tmp/jly/echi/data/pro/ce/'  # ae  蛋白质的特征向量
    file_name = "protein_fea_ce64_3阶.pkl"
    file_path = os.path.join(path, file_name)

    # 创建echi_neighbors的一阶和二阶的靶标向量
    num_1hop_target = 35  # ce
    num_2hop_target = 163  # ce
    # num_1hop_target = 61  # ae
    # num_1hop_target = 114  # cys
    echi_neighbors_emb_list = ppi_list(file_path,num_1hop_target,num_2hop_target)
    # print(echi_neighbors_emb_list[0])

    device = torch.device("cuda:0")
    echi = torch.zeros(echi_batch, dtype=torch.int).to(device)
    # 创建echi的嵌入向量
    echi_embedding = nn.Embedding(num_embeddings=10, embedding_dim=64).to(device)   # embedding_dim=64疾病特征维度
    echi_embeddings = echi_embedding(echi).to(device)
    # print(echi_embeddings)
    # print(echi_embeddings.size())  # 50*64  rensor

    # 聚合得到一阶和二阶的贡献值
    cell_i_list = _interaction_aggregation(echi_embeddings, echi_neighbors_emb_list)
    # print(cell_i_list)  # 50*64
    # 拼接聚合两个Isp的特征向量
    cell_embeddings1 = _aggregation(cell_i_list)
    # print(cell_embeddings1)  # 50*64
    return cell_embeddings1

# 获得疾病的一阶和二阶邻居靶标
def ppi_list(file_path,num_1_hop,num_2_hop):
    # 假设加载 pickle 文件得到名为 tensor 的二维张量
    with open(file_path, 'rb') as file:
        tensor = pickle.load(file)

    # 将张量转换为列表
    tensor_list = tensor.tolist()
    device = torch.device("cuda:0")
    # 创建两个列表分别存储靶标的特征向量
    target_list = [[], [],[]]

    # 将前35个靶标的特征向量存储在 target_list[0] 中，其余靶标的向量存储在 target_list[1] 中
    for i, target in enumerate(tensor_list):
        if i < num_1_hop:
            target_list[0].append(target)
        elif i > num_1_hop and i < num_2_hop:
            target_list[1].append(target)
        else:
            target_list[2].append(target)

    # tensor_0 = torch.tensor(target_list[0])
    # tensor_1 = torch.tensor(target_list[1])
    tensor_0 = torch.tensor(target_list[0]).to(device)
    tensor_1 = torch.tensor(target_list[1]).to(device)
    tensor_2 = torch.tensor(target_list[2]).to(device)
    target_list[0] = tensor_0
    target_list[1] = tensor_1
    target_list[2] = tensor_2
    return target_list
    # 现在 list_0 包含前35个靶标的特征向量，list_1 包含其余靶标的向量

def _interaction_aggregation(item_embeddings, neighbors_emb_list):  # 得到一阶和二阶的贡献即Isp
    interact_list = []
    n_hop = 3
    for hop in range(n_hop):
        # [batch_size, n_memory, dim]
        neighbor_emb = neighbors_emb_list[hop]
        # [batch_size, dim, 1]
        item_embeddings_expanded = torch.unsqueeze(item_embeddings, dim=2)  # 在第二维度上添加一个维度
        # [batch_size, n_memory]
        contributions = torch.squeeze(torch.matmul(neighbor_emb,
                                                   item_embeddings_expanded))  # 计算贡献值
        # [batch_size, n_memory]
        contributions_normalized = F.softmax(contributions, dim=1)  # 对贡献度进行 softmax 归一化
        # [batch_size, n_memory, 1]
        contributions_expaned = torch.unsqueeze(contributions_normalized, dim=2)  # 在第二维度上添加一个维度
        # [batch_size, dim]
        i = (neighbor_emb * contributions_expaned).sum(dim=1)  # 计算交互后的结果
        # update item_embeddings
        item_embeddings = i  # 更新当前节点的嵌入为交互后的结果
        interact_list.append(i)  # 将交互后的结果添加到列表中
    return interact_list  # 返回所有步骤的交互结果列表

def _aggregation(item_i_list):
    # [batch_size, n_hop+1, emb_dim]
    item_i_concat = torch.cat(item_i_list, 1)
    # print(item_i_concat.shape)
    # [batch_size, emb_dim]
    item_embeddings = aggregation_function(item_i_concat)
    return item_embeddings

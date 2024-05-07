import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import Node2Vec
# from node2vec import Node2Vec

 
def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def get_edge(cline_edge_file):
    cline_edge_list = []
    cline_edge = pd.read_csv(cline_edge_file, sep=',', header=0)
    cline_edge = np.array(cline_edge)
    cline_edge = cline_edge[:,1:652]
    for index,data in np.ndenumerate(cline_edge):
        if data == 1:
            cline_edge_list.append(index)
    cline_edge_list = list(map(list, zip(*cline_edge_list))) # 转置索引列表，然后将其转换为NumPy数组格式
    cline_edge_list = np.asarray(cline_edge_list)
    return torch.LongTensor(cline_edge_list)




if __name__ == '__main__':
    # ppi_edge_file = '/home/jly/echico/data3/pro/ppi_adj.csv' # CE
    # ppi_edge_file = '/home/jly/echico/data3/ae/pro/ppi_ae_adj.csv' # AE
    # ppi_edge_file = '/home/jly/echico/data3/cys/pro/ppi_cys_adj.csv' # CYS




    # ppi_edge_file = '/root/autodl-tmp/jly/GAECDS/data_pro/ppi_ae_adj.csv' # CYS
    # ppi_edge_file = '/root/autodl-tmp/jly/echi/data/pro/ppi_adj_ce.csv' # CYS



    ppi_edge_file = '/root/autodl-tmp/jly/echi/data/ce/adj3ce.csv' # CYS
    ppi_edge = get_edge(ppi_edge_file)
    node2vec_model = Node2Vec(ppi_edge, embedding_dim=64, walk_length=20,context_size=10, walks_per_node=10, num_negative_samples=1,sparse=True)


    cline_loader = node2vec_model.loader(batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(node2vec_model.parameters(), lr=0.01)
    for epoch in range(1, 101):
        loss = train(node2vec_model, optimizer, cline_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    protein_vec = node2vec_model()

    # 保存Tensor到文件
    # ce_ppi_emb_path = '/home/jly/echico/data3/pro/protein_fea.pkl'
    # ae_ppi_emb_path = '/home/jly/echico/data3/cys/pro/protein_fea.pkl'
    ae_ppi_emb_path = '/root/autodl-tmp/jly/echi/data/pro/ce/protein_fea_ce64_3阶.pkl'
    with open(ae_ppi_emb_path, 'wb') as file:
        pickle.dump(protein_vec, file)
    print(protein_vec)
   #  protein_vec = protein_vec.view(163 * 100)
    # 假设节点嵌入向量存储在变量 protein_vec 中
    print("节点嵌入向量的维度为:", protein_vec.shape)  # 163*64 torch.FloatTensor




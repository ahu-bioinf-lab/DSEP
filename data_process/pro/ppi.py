import pandas as pd
import networkx as nx
import numpy as np

# 读取ce蛋白质信息文件
proteins_data = pd.read_csv('/root/autodl-tmp/jly/echi/data/ce/一阶蛋白.csv')  # 假设蛋白质信息在名为 'protein_info.csv' 的文件中
# 读取蛋白质邻居信息文件
interactions_data = pd.read_csv('/root/autodl-tmp/jly/echi/data/ce/二阶pro.csv')  # 假设邻居信息在名为 'protein_neighbors.csv' 的文件中
# 三阶蛋白读取
three_data = pd.read_csv('/root/autodl-tmp/jly/echi/data/ce/三阶邻居.csv')


# # 读取AE蛋白质信息文件
# proteins_data = pd.read_csv('/home/jly/echico/data3/cys/pro/一阶pro.csv')  # 假设蛋白质信息在名为 'protein_info.csv' 的文件中
# # 读取蛋白质邻居信息文件
# interactions_data = pd.read_csv('/home/jly/echico/data3/cys/pro/二阶ppi.csv')


# 创建空图
G = nx.Graph()

# 添加蛋白质节点到图中
for index, protein in proteins_data.iterrows():
    id = protein['id']
    G.add_node(id)
    # print(len(G.nodes))

# 读取蛋白质一阶邻居信息文件


# 添加边到图中
for index, row in interactions_data.iterrows():
    id1 = row['id1']  # 假设邻居信息文件中有一列名为 Protein1
    id2 = row['id2']  # 假设邻居信息文件中有一列名为 Protein2
    G.add_edge(id1,id2)
    # G.add_edge(id2,id1)
    # print("现在的：",len(G.nodes))
for index, row in three_data.iterrows():
    id1 = row['id1']  # 假设邻居信息文件中有一列名为 Protein1
    id2 = row['id2']  # 假设邻居信息文件中有一列名为 Protein2
    G.add_edge(id1,id2)
    print("现在的：", len(G.nodes))
# 打印图的信息
# 获取图的边信息
edges = G.edges()
print(edges)
# 打印边信息
for edge in edges:
    print(edge)

# 获取图的节点列表，按照蛋白质的信息顺序排列
nodes_ordered = list(G.nodes)

# adj_matrix = nx.to_numpy_matrix(G,nodelist=nodes_ordered)
adj_matrix = nx.to_numpy_array(G, nodelist=nodes_ordered)
# 将邻接矩阵转换为 NumPy 数组（如果需要）
adj_array = np.array(adj_matrix)
# 打印邻接矩阵
# print(adj_array)

# 将邻接矩阵转换为 Pandas DataFrame
df = pd.DataFrame(adj_matrix,columns=nodes_ordered, index=nodes_ordered)
# 将 DataFrame 保存为 CSV 文件
# df.to_csv('/home/jly/echico/data3/pro/ppi_adj.csv', index=True, header=True)  # ce
df.to_csv('/root/autodl-tmp/jly/echi/data/ce/adj3ce.csv', index=True, header=True)  # ae



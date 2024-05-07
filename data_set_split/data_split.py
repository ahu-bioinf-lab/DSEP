import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import random
## 随机生成正负样本
# 读取 CSV 文件到 DataFrame
def data_split(data_path,random_seed=None,folder_name = None):
    #random.seed(42)
    # data_path = '/home/jly/echico/data3/echi/synergy.csv'  # 替换为你的 CSV 文件路径
    data = pd.read_csv(data_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间戳

    folder_name = os.path.join(folder_name, timestamp)
    print(folder_name)


    os.makedirs(folder_name)

    # 计算train和test的总数目比例为7:3
    train_ratio = 0.7
    test_ratio = 0.3
    total_samples = len(data)
    train_samples = round(total_samples * train_ratio)
    test_samples = round(total_samples * test_ratio)

    # 分离正负样本
    positive_samples = data[data['label'] == 1]
    negative_samples = data[data['label'] == 0]

    test_positive_count = round(len(positive_samples)/3)  # test中正样本的个数为总正样本数的1/3
    train_positive_count = round(len(positive_samples) * 2/3)   # train中正样本的个数为总正样本数的2/3

    # 将正样本放入train和test中
    train_positives = positive_samples.sample(train_positive_count,random_state=random_seed)
    test_positives = positive_samples.drop(train_positives.index).sample(test_positive_count,random_state=random_seed)

    # 计算train和test中负样本的个数
    train_negative_count = train_samples - train_positive_count
    test_negative_count = test_samples - test_positive_count

    # 从负样本中分别选取train和test的负样本
    train_negatives = negative_samples.sample(train_negative_count,random_state=random_seed)
    test_negatives = negative_samples.drop(train_negatives.index).sample(test_negative_count,random_state=random_seed)

    # 构建train和test数据集
    train_set = pd.concat([train_positives, train_negatives])
    test_set = pd.concat([test_positives, test_negatives])

    # 输出最终的train和test数据集大小
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    # 存储训练集和测试集至文件
    traindataset = f'{folder_name}/train_data.csv'
    testdataset = f'{folder_name}/test_data.csv'

    train_set.to_csv(traindataset, index=False)
    test_set.to_csv(testdataset, index=False)

    # 分别获取特征和标签列
    X_train = train_set[['smiles1', 'smiles2']]  # 替换为你的特征列名称
    y_train = train_set['label']  # 替换为你的标签列名称

    X_test = test_set[['smiles1', 'smiles2']]  # 替换为你的特征列名称
    y_test = test_set['label']  # 替换为你的标签列名称
    return testdataset,traindataset

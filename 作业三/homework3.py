import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

def preprocess_data(data_frame, is_training=True):
    data_list = np.array(data_frame[X_columns]).tolist()
    # 平均年龄
    age_mean = float(data_frame['Age'].mean())
    # 不同等级的费用中值
    cls_fare = dict(zip([1, 2, 3], [float(train.loc[train['Pclass']==p, 'Fare'].median()) for p in [1, 2, 3]]))
    for row in data_list:
        # 性别设为0或1
        row[1] = 0 if row[1] == 'male' else 1
        # 填充缺失的年龄
        row[2] = age_mean if type(row[2]) == float and math.isnan(row[2]) else row[2]
        # 处理费用
        row[5] = cls_fare[row[0]] if type(row[5]) == float and math.isnan(row[5]) else row[5]
        # 处理船舱编号
        row[6] = 0 if type(row[6]) ==float else ord(row[6][0]) - ord('A') + 1
        # 处理登船港口
        row[7] = 0 if type(row[7]) ==float else {'C': 1, 'Q': 2, 'S': 3}[row[7]]
    if is_training:
        labels = np.array(data_frame[Y_column]).tolist()
        return data_list, labels
    else:
        return data_list

def pca(data):
    '使用PCA将数据降到二维'
    pca_model = PCA(n_components=2)
    return pca_model.fit_transform(data)

def show_classification_result(data, label, title=''):
    '显示分类结果'
    cls_0 = data[label==0]
    cls_1 = data[label==1]
    plt.figure()
    plt.title(title)
    plt.scatter(cls_0[: ,0], cls_0[: ,1], label='class_0')
    plt.scatter(cls_1[: ,0], cls_1[:, 1], label='class_1')
    plt.legend()
    plt.show()

def save_predictions(data, file_path):
    '保存预测结果'
    start_id = 892
    with open(file_path, 'w') as f:
        f.writelines('PassengerId,Survived\n')
        for i, y in enumerate(data):
            f.writelines('{},{}\n'.format(i + start_id, int(y)))

if __name__ == '__main__':
    file1=("test.csv")
    file2=("train.csv")
    test=pd.read_csv(file1,low_memory=False)
    train=pd.read_csv(file2,low_memory=False)

    X_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
    Y_column = ['Survived']

    #预处理
    X_train, Y_train = preprocess_data(train)
    X_test = preprocess_data(test, is_training=False)

    #pca降维
    X_train_2_dim = pca(X_train)
    X_test_2_dim = pca(X_test)

    # 分类一：SVM
    svc = svm.SVC(kernel='linear')
    svc.fit(X_train, Y_train)
    predictions = svc.predict(X_test)
    #save_predictions(predictions, 'svm.csv')
    show_classification_result(X_test_2_dim, predictions, title='Linear SVM')

    # 分类二：决策树
    dtc = DecisionTreeClassifier(random_state=0, criterion='gini', max_leaf_nodes=10)
    dtc.fit(X_train, Y_train)
    predictions = dtc.predict(X_test)
    #save_predictions(predictions, 'decision_tree.csv')
    show_classification_result(X_test_2_dim, predictions, title='Decision Tree')

    # 聚类一：Kmeans
    clf = KMeans(n_clusters=2)
    clf.fit(X_train)
    predictions = clf.predict(X_test)
    #save_predictions(predictions, 'k_means.csv')
    show_classification_result(X_test_2_dim, predictions, title='K-means')

    #聚类二：hierarchical
    clf = AgglomerativeClustering(n_clusters=2)
    predictions = clf.fit_predict(X_test)
    #save_predictions(predictions, 'hierarchical.csv')
    show_classification_result(X_test_2_dim, predictions, title='Hierarchical')

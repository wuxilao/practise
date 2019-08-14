from sklearn import datasets
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    iris = datasets.load_iris()
    print(type(iris), dir(iris))

    x = iris.get('data')
    y = iris.get('target')

    # 随机划分训练集和测试集
    num = x.shape[0]  # 样本总数
    ratio = 7 / 3  # 划分比例，训练集数目:测试集数目
    num_test = int(num / (1 + ratio))  # 测试集样本数目
    num_train = num - num_test  # 训练集样本数目
    index = np.arange(num)  # 产生样本标号
    np.random.shuffle(index)  # 洗牌
    x_test = x[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
    y_test = y[index[:num_test]]
    x_train = x[index[num_test:], :]  # 剩余作为训练集
    y_train = y[index[num_test:]]

    clf_linear = svm.SVC(decision_function_shape="ovo", kernel="linear")
    clf_rbf = svm.SVC(decision_function_shape="ovo", kernel="rbf")
    clf_linear.fit(x_train, y_train)
    clf_rbf.fit(x_train, y_train)

    y_test_pre_linear = clf_linear.predict(x_test)
    y_test_pre_rbf = clf_rbf.predict(x_test)

    # 计算分类准确率
    acc_linear = sum(y_test_pre_linear == y_test) / num_test
    print('linear kernel: The accuracy is', acc_linear)
    acc_rbf = sum(y_test_pre_rbf == y_test) / num_test
    print('rbf kernel: The accuracy is', acc_rbf)

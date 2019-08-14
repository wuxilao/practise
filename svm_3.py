# ===============================样本不平衡、多分类的情况========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 创建不均衡样本
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
n_samples_3 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2), 0.5 * rng.randn(n_samples_2, 2) + [2, 2],0.5 * rng.randn(n_samples_3, 2) + [-3, 3]]  # 三类样本点中心为(1.5,1.5)、(2,2)、(-3,3)
y = [0] * (n_samples_1) + [1] * (n_samples_2)+ [2] * (n_samples_3)  # 前面的1000个为类别0，后面的100个为类别1，最后100个类别为2

# 创建模型获取分离超平面
clf = svm.SVC(decision_function_shape='ovo',kernel='linear', C=1.0)  # decision_function_shape='ovo'为使用1对1多分类处理。会创建n(n-1)/2个二分类。ovr为一对所有的处理方式
clf.fit(X, y)

# 多分类的情况下，获取其中二分类器的个数。
dec = clf.decision_function([[1.5,1.5]])  # decision_function()的功能：计算样本点到分割超平面的函数距离。 包含几个2分类器，就有几个函数距离。
print('二分类器个数：',dec.shape[1])

# 绘制，第一个二分类器的分割超平面
w = clf.coef_[0]
a = -w[0] / w[1]  # a可以理解为斜率
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]  # 二维坐标下的直线方程

# 使用类权重，获取分割超平面
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)


# 绘制 分割分割超平面
ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]  # 带权重的直线

# 绘制第一个二分类器的分割超平面和样本点
h0 = plt.plot(xx, yy, 'k-', label='no weights')
h1 = plt.plot(xx, wyy, 'k--', label='with weights')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()

plt.show()

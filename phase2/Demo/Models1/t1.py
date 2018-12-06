import seaborn as sns
import numpy as np
from numpy.random import randn
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

# style set 这里只是一些简单的style设置
sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5)})
np.random.seed(1425)

data = randn(7500)
# plt.hist(data)
# hist 其它参数
# x = stats.gamma(3).rvs(5000)
#plt.hist(x, bins=80) # 每个bins都有分界线
# 若想让图形更连续化 (去除中间bins线) 用histtype参数
# plt.hist(x, bins=80, histtype="stepfilled", alpha=.8)
# sns.distplot(x, rug=True, hist=False)


# 利用kdeplot来确定两个sample data 是否来自于同一总体
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
c1, c2, c3 = sns.color_palette('Set1', 3)

dist1=stats.norm(60,20).rvs(500)
dist2=stats.norm(85,20).rvs(500)
dist1=[random.uniform(50,60) for i in range(200)]+list(dist1)+[random.uniform(60,90) for i in range(200)]
dist2=[random.uniform(60,90) for i in range(200)]+list(dist2)+[random.uniform(101,250) for i in range(200)]
# dist1, dist2是两个近似正态数据, 拥有相同的中心和摆动程度
sns.kdeplot(dist1, shade=True, label='open domain')
sns.kdeplot(dist2, shade=True,  label='biomedical')

# # dist3 分布3 是另一个近正态数据, 不过中心为2.
# sns.kdeplot(dist1, shade=True, color=c2, ax=ax2)
# sns.kdeplot(dist3, shade=True, color=c3, ax=ax2)

# # vertical 参数 把刚才的图形旋转90度
# plt.figure(figsize=(4, 8))
# data = stats.norm(0, 1).rvs((3, 100)) + np.arange(3)[:, None]
#
# with sns.color_palette("Set2"):
#     for d, label in zip(data, list("ABC")):
#         sns.kdeplot(d,  shade=True, label=label)

# plt.hist(data, vertical=True)
# error vertical不是每个函数都具有的
plt.show()





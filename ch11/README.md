# 处理无标记的数据-聚类分析(Clustering Analysis)

目标是找到一个数据的自然分组使得相类似的东西分到不同的聚类。

- 使用流行的k-means算法搜寻相似性的中心点
- 从下往上生成层级的聚类树
- 使用基于密度的聚类方法识别任意的对象形状(Identifying arbitrary shapes of objects using a density-based clustering approach)

## 使用k-means根据相似性组合对象

k-means算法是一种基于原型的聚类算法(prototype-based clustering)。而基于运行的聚类则指的是聚类都可以用一个原型表示，可以是根据相似点的中心(centroid, average)，特征是连续值的情况；还有基于最具代表性或最多出现点的分类特征情况。

具体使用时，我们需要先验的给定一个k值，而k值的选择非常影响算法性能。

经典k-means算法的步骤：

1. 从样本点中随机选取k个中心(centroid)作为初始的聚类中心点(center)
1. 指定每个样本点到最近的中心(centroid) $\miu^{(j)},j\inc\{1,...,k\}$
1. 移动中心(centroid)到指定给此中心的样本中心点(center)
1. 重复第2~3步，直到聚类分配不再变化，或者达到了用户定义的容差范围内，或达到了最大的迭代次数

如何测量对象之间的相似性？

对于连续值特征的情况，可以用平方欧几里得距离(squared Euclidean distance)的大小作为相似性判断依据。

聚类惯量(cluster inertia)：在聚类内的平方误差和(within-cluster sum of squared errors, SSE)。最小化聚类惯量的优化问题。

k-means++：

1. 初始化一个空集M来存储k个中心
1. 从输入样本中随机选择第一个中心并加入到集合M中
1. 对于不在M中的样本，找到与M中中心的最小平方距离
1. 使用一个带权重的概率分布(样本到M的距离作为权重)随机选择下一个中心
1. 重复第2、3步直到k个中心都完成选择
1. 执行经典的k-meams算法

k-means有个比较大的缺点是需要一开始指定k个中心，而可能其中的中心周围是没有样本的。

## 硬聚类和软聚类(Hard versus soft clustering)

硬聚类只每个样本都仅分配给一个聚类；软聚类允许一个样本分配给多个聚类。

FCM算法(fuzzy C-means, soft k-means)是一种软聚类算法，样本若在多个聚类中，则针对每个聚类分配一个概率。

## 使用肘方法(elbow method)定位最优聚类个数

k-means聚类算法中会给出within-cluster SSE(distortion)，即：`km.inertia_`。选取不同的聚类个数，根据聚类个数与distortion的图像即可找到最优的聚类个数值。

## 通过silhouette图像(silhouette plots)量化聚类的质量

silhouette分析可以用来评估聚类的质量，它不仅仅适用于k-means算法，其他聚类算法同样适用。

1. 计算聚类粘度(cluster cohesion)$a^{(i)}$：一个样本与所有其他在同一个聚类的距离平均值
1. 计算与下一个最近的聚类的聚类隔离(cluster separation)$b^{(i)}$：一个样本与所有最近聚类的距离平均值
1. 计算silhouette的值$s^{(i)}$：聚类粘度与隔离之差，并除以两个的最大值

## 组织聚类为层级树(hierarchical tree)

使用层级聚类算法，可以容易的做出系统树图(dendrograms)，从而容易做出有意义的分类解释(taxonomies)。而且我们不需要提前给出聚类的个数。

- 凝聚层级聚类算法(agglomerative hierarchical clustering)：一开始每个样本都作为独立的聚类，不断合并最近的聚类对直到剩下一个聚类
- 分裂层级聚类算法(divisive hierarchical clustering)：一开始所有样本都在一个聚类中，递归分裂聚类为小的聚类知道所有聚类仅仅包含一个样本

凝聚层级聚类的两个标准算法：

- 单一连接(single linkage)：计算没两个聚类中最相似的成员距离，合并其中距离最小的两个聚类
- 完全连接(complete linkage)：与单一连接类似，只不过比较的是最不相似的成员来执行合并。

以完全连接为例：

1. 计算所有样本的距离矩阵
1. 每个数据点都作为一个聚类
1. 基于最远的成员距离合并最近的两个聚类
1. 更新距离矩阵
1. 重复步骤2~4直到仅剩下一个聚类

## 通过DBSCAN定位高密度区域

基于密度的噪声应用空间聚类(Density-based Spatial Clustering of Application with Noise, DBSCAN)

- 核心点(core point)：局部区域(MinPts)的中心
- 边界点(border point)：比MinPts具有较少邻居的点，但位于核心点的邻域内
- 噪声点(noise point)：既不是核心点，也不是边界点的点

其中MinPts表示组成区域的最少点数。

1. 对于每个核心点形成一个隔离的聚类，或者形成一个核心点的连接组(核心点是被连接的，若它们都位于核心点的邻域内)
1. 指派每个边界点给相应核心点的聚类

DBSCAN算法的一个优点是它不需要假设聚类像k-means算法中的有一个球形的形状。
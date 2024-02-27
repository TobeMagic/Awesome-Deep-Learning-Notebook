# 深度学习前言

深度学习算法是一类基于生物学对人脑进一步认识，将神经-中枢-大脑的工作原理设计成一个不断迭代、不断抽象的过程，以便得到最优数据特征表示的机器学习算法；该算法从原始信号开始，先做**低层抽象**，然后逐渐向**高层抽象迭代**，由此组成深度学习算法的基本框架。

> AI发展困难点

在AI发展过程中遇到的几大困难：

1）机器需要学习/理解世界的运行规律（包括物理世界、数字世界、人……以获得一定程度的常识）

2）机器需要学习大量背景知识（通过观察和行动）

3）机器需要理解世界的状态（以做出精准的预测和计划）

4）机器需要更新并记住对世界状态的估测（关注重大事件，记住相关事件）

5）机器需要逻辑分析和规划（预测哪些行为能让世界达到目标状态）

> 其中最大挑战之一就是**如何让机器拥有常识**——即让机器获得填充空白的能力。比如“John背起包来，离开了房间”，由于人类具备常识，因此我们能够知道John在这个过程中需要站起来，打开房门，走出去——他不会是躺着出去的，也不会从关着的门出去，可机器并不知道这一点。又或者我们即使只看到了半张人脸也能认出那人是谁，因为人类常识里左右脸都是通常长得差不多，但机器同样不具备这种能力。

对于一个AI系统来说，**预测+规划=逻辑分析（Predicting + Planning = Reasoning）**。如果想要让机器能够了解并且预测世界的规律，有以下几种方案。

1. 强化学习（Reinforcement Learning）需要建立一个世界模拟器（World Simulator），模拟真实世界的逻辑、原理、物理定律等。不过真实世界太过复杂，存在大量的表征学习参数，使得机器学习的计算量相当冗余，这在有限的时间内无法学习到成千上亿的参数，需要耗费大量的资源。
2. 无监督学习需要机器处理大量没有标记的数据，就像给它一堆狗的照片，却不告诉它这是一条狗。机器需要自己找到区分不同数据子集、集群、或者相似图像的办法，有点像婴儿学习世界的方式。
3. 无监督/预测学习可以让机器获得常识，但现在我们常用的监督学习并做不到这一点。从本质上来说，在无监督学习方面，生物大脑远好于我们的模型。

# 核心模型架构

下面是核心的深度学习基础模型，其他模型都是基于此叠加构建的

| 模型名称                                           | 主要特点和解决问题                                           | 缺点                                                         |
| :------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 多层感知机（Multilayer Perceptron, MLP）           | - 多层感知机是最基本的前馈神经网络模型。<br>- 通过使用多个隐藏层和非线性激活函数，它可以对复杂的非线性关系建模。<br>- 适用于分类和回归问题。 | - 容易过拟合，需要适当的正则化和调参来避免。<br>- 对于高维稀疏数据，效果可能不佳。<br>- 对输入数据的缩放敏感。 |
| 卷积神经网络（Convolutional Neural Networks, CNN） | - CNN适用于处理图像和计算机视觉任务。<br>- 利用卷积层和池化层来捕捉空间结构和局部特征。<br>- 具有参数共享和平移不变性的特性，减少了模型的参数量。<br>- 在许多计算机视觉任务中表现出色，如图像分类、目标检测和图像分割。 | - 对于长距离依赖性的任务，如序列建模，CNN可能不是最佳选择。<br>- 对于输入尺寸变化较大的问题，需要适应性池化或其他方法来处理不同尺寸的输入。 |
| 循环神经网络（Recurrent Neural Networks, RNN）     | - RNN适用于序列数据建模，如文本和语音。<br>- 具有自连接的隐藏层，可以捕获序列中的时间依赖关系。<br>- 可以处理可变长度的输入序列。<br>- 长短期记忆网络（Long Short-Term Memory, LSTM）和门控循环单元（Gated Recurrent Unit, GRU）是常用的RNN变体。 | - RNN在处理长期依赖性时容易出现梯度消失或梯度爆炸问题。<br>- 训练过程较慢，很难并行化。<br>- 对于较长的序列，内部记忆状态可能会受限制。 |
|                                                    |                                                              |                                                              |

## 基础核心模型

### 感知器 (Perceptron) & MLP-BP神经网络 （全连接)

可以先看一个非常有趣的讲解 （**感知器是一种单层神经网络，而多层感知器则称为神经网络。**）： https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53  

#### 感知器

感知器是神经网络的 Fundamentals，其其实可以作为机器学习的范畴，因为其也算是浅度学习，但是感知器作为深度学习最基本的神经元单元入门，所以放入在当前章节。

在1977年由Frank Roseblatt 所发明的感知器是最简单的ANN架构之一（**线性函数加上硬阈值**（算是广义线性模型），**这里阈值不一定是0**），受在一开始的生物神经元模型启发（`XOR`**问题逻辑问题**），称之为阈值逻辑单元（TLU，threshold logistic unit)  或线性阈	值单元（LTU,linear threshold unit)，其是一个**使用阶跃函数的神经元**来计算，可被用于线性可分二分类任务，也可设置多个感知器输出实现多输出分类以输出n个二进制结果（缺点是各种类别关系无法学习），一般来说还会添加一个偏置特征1来增加模型灵活性。

![图像](core algorithm.assets/FXo8u7JaQAANoNm.png)

> 在感知器中引入一个偏置特征神经元1的目的是为了增加模型的灵活性和表达能力。这个偏置特征对应于一个固定且始终为1的输入，**其对应的权重称为偏置项（bias）。通过调整偏置项的权重，我们可以控制 TLU 的决策边界在特征空间中平移或倾斜**。（正常来说的话，这个偏置项都是在每个神经元当中所存在，而不是作为单独一个输入存在，能更灵活）
>
> > 在感知器中，将偏置特征固定为1的选择是为了方便计算和表示。
> >
> > 当我们引入一个偏置特征时，可以将其视为与其他输入特征一样的维度，并赋予它一个固定的值1。这样做有以下几个好处：
> >
> > 1. 方便计算：将偏置项乘以1相当于**直接使用权重来表示该偏置项**。在进行加权求和并应用阈值函数时，不需要额外操作或考虑。
> > 2. 参数统一性：通过将偏置项作为一个**独立的权重**进行处理，使得所有输入特征（包括原始输入和偏置）具有**相同的形式和统一性**。
> > 3. 简洁明了：固定为1的偏置特征能够**简化模型参数表示**，并使其更易理解和解释。
> >
> > 请注意，在实际应用中，对于某些任务
> > 可能需要调整默认值1以适应数据分布或优化模型性能。但基本原则仍然是保持一个常数值作为额外输入特征，并且通常会根据具体情况对其进行学习或调整。
>
> 具体来说，引入偏置特征1有以下几个原因： 
>
> 1. **平移决策边界**：通过调整偏置项的权重，可以使得决策边界沿着不同方向平移。如果没有偏置项，则决策边界将必须过原点(0, 0)。
>
> 2. 控制输出截距：当所有**其他输入都为零时，只有存在偏置项才能使感知器产生非零输出**。
>
> 3. 增强模型表达能力：引入一个额外维度后，在某些情况下会更容易找到**合适分割样本空间线性超平面位置**。
>
>    总之，在感知器中引入偏置特征1可以使模型更加灵活，能够适应不同的决策边界位置，并增加了模型对输入数据的表达能力。

其中，Siegrid Lowel非常著名的一句话“一同激活的神经元联系在一起”（Hebb的思想，一个生物元经常触发另外一个神经元，二者关系增强），故此Rosenblatt基于该规则提出一种**感知器训练算法**，其加强了有助于减少错误的连接，如果预测错了，比如预测目标是1，预测到0，就会增强对应神经元权重和偏置，如果预测目标是0，预测到1，就会减小。（根据阶跃函数性质值越大为1，值小为0）

以下是感知器训练算法的步骤（只有一层神经网络）：

1. 初始化参数：初始化权重向量 w 和偏置 b 为零或者随机小数。（一般来说感知器个数不多情况下，个数多则可以使用如神经网络的初始化如He初始化等）
2. 对每个训练样本进行迭代：
   - 计算预测输出 **y_hat = sign(w * x + b)**，其中 w 是权重向量，x 是输入特征向量，b 是偏置项，并且 sign() 函数表示取符号（正负，二分类为例）。
   - 更新权重和偏置：
      - 如果 **y_hat 等于实际标签 y，则无需更新参数**。
      - 如果 y_hat 不等于实际标签 y，则根据下面的规则更新参数：（注意这里的更新规则是目标是1，预测结果为0则权重变小，否则变大，已符合sigmoid激励函数）
         - 权重更新规则：w = w + η * (y - y_hat) * x，其中 η 是学习率（控制每次更新的步长）。
         - 偏置更新规则：b = b + η * (y - y_hat)。(偏移)

这个过程会不断迭代直到所有样本被正确分类或达到预定的停止条件（如达到最大迭代次数）。从以下我们就可以看到线性可分的感知机训练过程和线性不可分的感知机训练过程，在线性不可分的情况下，泛化能力较差。

![img](classical algorithm.assets/20190112221655864.gif)

![img](classical algorithm.assets/20190112221824187.gif)

#####  鸢尾花多分类案例

我们以鸢尾花多分类作为案例讲解

Sci-learn:https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

Wikipedia:https://en.wikipedia.org/wiki/Iris_flower_data_set

<img src="core algorithm.assets/image-20230817123141139.png" alt="image-20230817123141139" style="zoom:33%;" />

<img src="core algorithm.assets/image-20230817123152718.png" alt="image-20230817123152718" style="zoom:33%;" />

我们从以上的可视化就可以知道，**用Perceptorn分类必然效果不好，因为其线性不可分**。

**不使用库**实现感知器**一对多策略多分类**鸢尾花数据集任务的代码：

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:
    """
    设计架构
    1. 初始化基本超参数
    2. 根据算法模型抽象化权重训练流程
    3. 训练中细分单个样本训练和预测中细分单个样本预测以实现多样本训练和预测 """
    def __init__(self, learning_rate=0.1, num_epochs=20):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(self, X, y):
        # 添加偏置项到输入数据中
        X = np.insert(X, 0, 1, axis=1)

        # 初始化权重为随机值
        np.random.seed(42)
        self.weights = []

        # 训练模型（每个类别都有自己独立的感知器）
        for class_label in set(y): # 集合去重
            binary_labels = np.where(y == class_label, 1, -1) # True is 1 or False is  -1
#             print(binary_labels)
            weights_class = self.train_single_perceptron(X, binary_labels)
            self.weights.append(weights_class)

    def train_single_perceptron(self, X, y):
        weights = np.random.rand(X.shape[1]) # 随机初始化后训练（每个样本的特征数）
        for _ in range(self.num_epochs): #轮次
            for i in range(len(X)):
                prediction = self.predict_single_sample(X[i], weights) # 数据和权重求解
                error = y[i]-prediction
                # 更新权重
                update = self.learning_rate*error*X[i]
                weights += update
        return weights

    def predict_single_sample(self, x, weights):
        """receive x and weights return step function"""
        activation_value = np.dot(x, weights)
        return 1 if activation_value >= 0 else -1 # step function (corressponds to the previous binary_labels)

    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1) # 同样需要插入偏置神经元1
        predictions = []
        for i in range(len(X_test)):
            class_predictions = []
            for perceptron_weights in self.weights:
                class_predictions.append(self.predict_single_sample(X_test[i], perceptron_weights))
            predicted_class = np.argmax(class_predictions) # 如果一样大返回最先的下标
#             print(class_predictions) 
#             print(predicted_class)
            predictions.append(predicted_class)
        return predictions


# 加载鸢尾花数据集（数据顺序排列，一定要打乱，泛化能力）
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42) # 
# X_train, X_test, y_train, y_test = data.data[:120,:],data.data[120:,:],data.target[:120],data.target[120:] # , random_state=42
# 创建感知器对象并训练模型
perceptron = Perceptron()
perceptron.train(X_train, y_train)

# 使用测试数据集进行预测
predictions = perceptron.predict(X_test)
print(np.array(predictions))
print(y_test)
# print(type(y_test))

accuary = sum(predictions == y_test)/len(y_test) 
accuary = accuracy_score(y_test,predictions)
print(accuary)
```

输出

```
[1 0 1 0 1 0 0 2 1 1 2 0 0 0 0 0 2 1 1 2 0 2 0 2 2 2 1 2 0 0]
[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
0.8333333333333334
```

**使用sklearn库**实现感知器分类鸢尾花数据集任务的代码 (perceptron 其实还是算是浅度学习，所以在sklearn中）：

```python
from sklearn.linear_model import Perceptron

# 加载鸢尾花数据集
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42) # 随机数一样的话，随机结果是一样的
# data.data[:120,:],data.data[120:,:],data.target[:120],data.target[120:] #

# 创建并训练感知器模型
perceptron = Perceptron(eta0=0.1, max_iter=100)
perceptron.fit(X_train, y_train)

# 使用测试数据集进行预测
predictions = perceptron.predict(X_test)
print(predictions)
print(y_test)

accuary = sum(predictions == y_test)/len(y_test)
print(accuary)
```

输出：

```
[1 0 2 0 1 0 0 2 1 0 2 0 0 0 0 0 2 0 0 2 0 2 0 2 2 2 2 2 0 0]
[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
0.8
```

>  `sklearn.linear_model.Perceptron`的参数：
>
>  - `penalty`: 惩罚项（默认值：None）。可以选择"l1"或"l2"来应用L1或L2正则化，也可以选择None不应用任何惩罚项。
>
>  - `alpha`: 正则化强度（默认值：0.0001）。较大的alpha表示更强的正则化。
>
>  - `fit_intercept`: 是否拟合截距（默认值：True）。如果设置为False，则模型将不会拟合截距。
>
>  - `max_iter`: 最大迭代次数（默认值：1000）。指定在达到收敛之前要执行的最大迭代次数。
>
>  - `tol`: 收敛容忍度（默认值：1e-3）。指定停止训练时目标函数改善小于该阈值时的容忍程度。
>
>  - `shuffle`: 是否在每个周期重新打乱数据顺序（默认值：True）。
>
>  - `eta0`: 初始学习率（默认值：1.0）。控制权重更新速度的学习率。较低的初始学习率可能有助于稳定模型收敛过程，但训练时间可能变长。
>
>  - `random_state`: 随机种子。提供一个整数以保证结果可重复性，或者设置为None以使用随机状态。
>
>  - `verbose`: 是否打印详细输出（默认值：0）。设置为1时，会定期打印出损失函数的值。
>

在这两个例子中，我们都使用了鸢尾花数据集，并将其分为训练和测试数据。然后，我们创建了一个感知器对象（自定义或Scikit-Learn提供的），并使用`train()`方法（自定义）或`fit()`方法（Scikit-Learn）来训练模型。最后，在测试数据上使用`predict()`方法来生成预测结果。（其中我们还可以设置一些超参数达到优化的目的）

>  扩展：
>
>  `MLPClassifier`和Keras中的`Dense`层都用于实现多层感知器（Multi-Layer Perceptron）模型。
>
>  在Scikit-Learn库中，`MLPClassifier`是一个基于神经网络的分类器，它使用反向传播算法进行训练，并可以处理多类别分类问题。你可以通过指定不同的参数来配置隐藏层、激活函数、优化算法等。
>
>  而在Keras库中，`Dense`层也被用作构建神经网络模型的一部分。它定义了全连接层（fully connected layer），其中每个输入节点与输出节点之间都有权重连接。你可以通过设置不同的参数来调整该层的大小、激活函数等。
>
>  虽然两者具有相似的功能，但由于框架和接口不同，它们在代码编写上可能会有所差异。因此，在使用时需要根据所选框架来适当调整代码。
>
>  总体上说，**“MLPClassifier”和Keras中“Dense”层都是为了实现多层感知器模型而设计的工具，在不同框架下提供了类似功能但语法略有差异。**

##### 应用场景

相比其他机器学习算法，感知器具有以下优势：

1. 简单而高效：感知器算法**非常简单且易于实现，计算速度快**。
2. 对噪声数据鲁棒：由于**其使用了阶跃函数作为激活函数，在处理带有噪声数据时表现较好**。
3. 支持在线学习：感知器是一种在线学习算法，可以逐步更新权重和阈值，并在每次迭代中对新样本进行训练。

然而，感知器也存在一些局限性：

1. **仅适用于线性可分问题**：由于其基于线性模型，在**处理非线性可分问题时无法取得良好的结果**。
2. **只能进行二分类**：感知器只能用于二分类任务，并**不能直接扩展**到多类别分类问题上。
3. **对输入特征缩放敏感**：感知器对输入特征的缩放比较敏感，如果**特征之间的尺度差异较大**（因为结果是根据值的大小决定的，所以在使用前需要数据特征归一化或者标准化），可能会影响算法的性能。

在实际应用中，当面对非线性可分问题时，可以考虑使用其他更复杂的模型，如支持向量机、神经网络等。这些模型具有更强大的表示能力，并且能够处理更为复杂和抽象的关系。然而，在某些简单问题上，感知器仍然是一个有效且高效的选择。

总结起来就是，感知器适用于解决线性可分二分类问题，并且**具有简单、高效和鲁棒等优点**。但它无法处理非线性可分问题，并且只能进行二分类任务。对于不同类型或更复杂的问题，可以考虑使用其他更适合的方法。

#### MLP全连接神经网络

`BP神经网络`，指的是用了**“BP算法”进行训练的“多层感知器模型”（MLP)。**并为了TLU感知机算法正常工 作，对MLP的架构进行了修改，即将阶跃函数替换成其他激活函数，如`tanh`，`Relu`。这里之所以用反向传播是因为多层的感知机无法再用感知机学习规则来训练. 

##### 原理概述

`layers.Dense`是`Keras`中的一种常用层类型，它实现了全连接层（Fully Connected Layer），也叫稠密层（Dense Layer）或者仿射层（Affine Layer）。该层的作用是将前一层的所有节点都连接到输出层的每个节点上，因此它可以将前一层的输出转化为特征向量。

`layers.Dense`的实现原理是利用**矩阵乘法，将输入数据（一般是一个向量）与权重矩阵相乘，然后加上偏置项，最后通过激活函数得到输出结果**。其数学表达式如下：

$y = f(Wx + b)$

其中，$x$是输入向量，$W$是权重矩阵，$b$是偏置项，$f$是激活函数，$y$是输出向量。	

##### 输入与输出

神经元的输入并不一定需要经过Flatten层。Flatten层的作用是将输入数据展平为一维向量，通常在连接全连接(Dense)层之前使用，**以便适应全连接层的输入要求**。

对于一些**具有固定尺寸的输入数据，例如图像数据，可以直接作为多维张量传递给Dense层**，而无需使用Flatten层。例如，对于28x28像素的灰度图像数据，可以将其表示为（28，28，1）的三维张量，其中1表示通道数（灰度图像只有一个通道），然后直接连接到Dense层。

然而，对于一些**不具有固定尺寸的输入数据**，如不定长的序列数据（例如文本或时间序列数据），通常需要使用Flatten层将其转换为固定长度的向量，然后再传递给Dense层进行处理。

因此，是否需要使用Flatten层取决于你的输入数据的形状和你希望构建的网络结构。如果输入数据已经是适合Dense层的形状，就不需要使用Flatten层。

##### Timedistributed(Dense)

Timedistributed(Dense) 是一种在神经网络中常用的层级扩展技术，它用于处理时间序列数据或具有时间维度的序列数据。在这个技术中，**Dense（全连接）层被应用到每个时间步上，以实现对整个序列的建模**。 其实该方法相比常规的全连接Dense层是少了很多的参数，因为dense对每一刻数据有一个参数，层输出后会丢失整体的时间序列特征，而timedistributed 则是不对时间序列用特征，对每一刻的时间步的特征进行参数拟合，而该参数对每一个时间步都是共享的，假如说我们每一个时间步只有一个特征，那该层只是对数据进行放大或变小变换，如果每一个时间步有两个特征，则只是不同特征的变换尺度不同，每一个时间步的特征参数是一致的，就可以很好保留了**时间序列**特征。

**原理和推导过程：**

在处理时间序列数据时，传统的Dense层只能对整个序列进行一次性的操作，无法捕捉到序列中每个时间步之间的时序关系。为了解决这个问题，引入了Timedistributed(Dense)层。

假设我们有一个输入张量X，形状为(batch_size, time_steps, input_dim)，其中batch_size表示批量大小，time_steps表示时间步数，input_dim表示输入维度。而Timedistributed(Dense)层的输出为(batch_size, time_steps, units)，其中units表示输出维度。

具体推导过程如下：

1. 假设输入张量X经过Timedistributed(Dense)层后的输出为Y。
2. 对于每个时间步t，输入张量X的第t个时间步的片段为X[:, t, :]，形状为(batch_size, input_dim)。
3. 对于每个时间步t，Timedistributed(Dense)层将输入片段X[:, t, :]与共享的权重W和偏置b结合，进行全连接操作，得到输出片段Y[:, t, :]，形状为(batch_size, units)。
4. 重复上述步骤，对每个时间步t都执行相同的全连接操作，从而得到整个序列的输出张量Y。

通过Timedistributed(Dense)层，我们可以**在每个时间步上应用相同的权重和偏置**，使得模型能够对时间序列数据进行逐步建模，更好地捕捉到时序关系。 

**案例模板代码：**

以下是一个基于Keras库的Python代码示例，展示了如何使用Timedistributed(Dense)层构建一个简单的循环神经网络（RNN）模型，并将其应用于时间序列数据的分类任务。请注意，这只是一个模板，您可以根据自己的需求进行调整和扩展。

```python
from keras.models import Sequential
from keras.layers import Timedistributed, Dense, LSTM

# 构建模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, input_dim)))
model.add(Timedistributed(Dense(units=32, activation='relu')))
model.add(Dense(units=num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val))

# 使用模型进行预测
y_pred = model.predict(X_test)
```

在上述代码中，我们首先构建了一个Sequential模型。然后，将LSTM层添加到模型中，设置return_sequences=True以保留每个时间步的输出。接下来，我们添加了Timedistributed(Dense)层，其中units参数表示输出维度。最后，我们添加了一个全连接层(Dense)进行最终分类，并编译模型以进行训练和评估。

### 卷积神经网络

CNN非常强大！！ 跳出框框思考！使用一维CNN对表格数据进行特征提取。或者使用DeepInsight（一种将非图像数据转换为图像以用于卷积神经网络架构的方法），将表格数据转换为图像，利用 CNN 的优势。

推荐视频：https://www.youtube.com/watch?v=KuXjwB4LzSA&list=RDCMUCYO_jab_esuFRV4b17AJtAw

确定卷积层的最佳数量以及它们对其他参数的影响是一个挑战性的任务，通常需要进行实验和调整来找到最佳的模型架构。

一般而言，卷积层数量的选择可以基于以下因素进行评估：

1. **数据集大小**和复杂程度：更大更复杂的数据集可能需要更深的卷积神经网络，以便提取更丰富的特征。较小的数据集则可能需要较浅的卷积神经网络，以避免过拟合。

2. 训练时长和计算资源：更深的卷积神经网络需要更长的训练时间和更多的计算资源。在限制时间和计算资源的情况下，可能需要权衡深度和精度。

3. 预训练模型的可用性：使用预训练模型可以减少训练时间并提高模型的精度。如果可用的预训练模型包含了与问题相关的卷积层，则可以考虑从这些层开始，然后通过微调来逐步优化模型。

除了卷积层的数量外，其他参数也会影响模型的性能。例如，卷积层的大小、步幅、填充等参数会影响特征图的大小和数量。池化层的类型、大小和步幅也会影响特征图的大小和数量，从而影响后续层的表现。因此，在设计卷积神经网络时，需要综合考虑这些参数，并进行实验和调整以找到最佳的模型结构。

> Q: CNN 1D与1至5个卷积层模型测试性能获得的准确性分别为88.36%、89.48%、88.86%、87.96和86.89%。五个1D CNN层是最大的界限，因为这个层上的函数图的最小尺寸已经超过了。
>
> A: 这个问题涉及到卷积神经网络中的**感受野（receptive field）概念**。
>
> 在卷积神经网络中，每一层的卷积核实际上是对上一层特征图的局部区域进行处理，而**该局部区域的大小就是该层的感受野大小**。这意味着，随着层数的增加，感受野也会逐渐扩大。
>
> 在1D CNN中，每个卷积核只能查看其左右固定数目的元素，这个固定数目就是感受野。因此，通过堆叠多个1D CNN层，可以使得后面的层拥有更大的感受野，从而提取更全局的特征。
>
> 但是，当1D CNN层数过多时，每一层的输出的长度也会逐步缩小。这是因为，在1D CNN中，卷积操作将输入向量的每个元素映射到输出向量的一个元素，因此每次卷积操作都会减少向量长度。随着层数的增加，输出向量的长度也会逐渐缩小，最终可能会导致信息丢失，从而影响模型性能。
>
> 因此，作者在该问题中使用了1至5个1D CNN层进行测试，并发现5层是极限。作者指出，当使用5个1D CNN层时，**最后一层的输出长度已经非常短，无法再添加更多的卷积层**。因此，**作者不能通过增加层数来进一步提高模型性能，而必须尝试其他方法，如调整卷积核大小、池化方式等**，以达到更好的性能。或者使用transformer模型思想，一层就可以看到整个序列的感受野。

> 卷积（Convolution）这个名词最初来源于数学领域，指的是两个函数之间的一种数学运算，也称为函数的乘积积分。在深度学习中，卷积操作是通过将一个输入信号与一个卷积核进行卷积运算来提取特征。在这个过程中，**卷积核会在输入信号上滑动，并在每个位置进行一次乘积累加的计算**，最终得到一个输出特征图。因此，这个操作被称为卷积。
>
> 在深入了解卷积神经网络（Convolutional Neural Network, CNN）的原理之前，让我们使用一个简单的生活例子来说明其工作原理。想象一下你正在观看一部电影，而电影是由连续的图像帧组成的。你想要识别电影中的主要角色。这时，你的大脑就会使用类似于卷积神经网络的机制进行处理。首先，你的大脑会将图像帧传递给视觉皮层（Visual Cortex），这相当于CNN中的输入层。在视觉皮层中，一系列的神经元会对图像进行处理，每个神经元对应一个特定的区域（感受野）。然后，每个感受野会执行一个局部感知操作，类似于CNN中的卷积操作。这个操作类似于你的眼睛聚焦在图像的一个小部分，并提取特定的特征。例如，某个感受野可能会注意到脸部特征，而另一个感受野可能会注意到物体的纹理。接下来，提取的特征会通过神经元之间的连接进行传递，这类似于CNN中的池化操作。在池化过程中，一组相邻的感受野的特征被合并为一个单一的特征。这样做可以减少数据的维度，并提取更加重要的特征。这些特征将继续传递到更高级别的层次，类似于CNN中的隐藏层。在这些层次中，神经元将学习更加抽象和高级的特征表示，例如面部表情、物体形状等。最终，通过一系列的卷积、池化和隐藏层的操作，网络可以学习到适合于图像识别的特征。这些特征将传递到输出层，用于识别电影中的主要角色。
>
> 总的来说你的大脑类似于一个卷积神经网络。它通过局部感知、特征提取和特征学习的方式，从连续的图像帧中识别出主要角色。卷积神经网络的原理与此类似，通过卷积、池化和隐藏层的操作，从输入图像中提取有用的特征，并用于各种图像处理任务，如图像分类、目标检测等。尽管实际的卷积神经网络可能更复杂，包含更多的层和参数，但它们都遵循类似的原理

注意点：一定要知道一维卷积、二维卷积、三维卷积不同的是方向上的卷积，并且要知道一维卷积如何处理二维/三维数据，二维卷积如何处理三维数据。

####Conv1D

我们考虑一个简单的情况，就像处理时间序列数据一样。想象你正在观察某个城市在一周内的每日气温变化。你想要通过一维卷积来平滑这些数据，以便更好地理解气温趋势（在该例子其实就是三个连续数值不同加权求和得到一个代表性的数值）。

假设你有一周的气温数据，表示为一维数组：

```python
temperature_data = [20, 22, 24, 25, 28, 27, 26]
```

现在，让我们使用一个长度为3的一维卷积核（或过滤器）来对这些数据进行卷积操作。假设卷积核为：

```python
kernel = [0.5, 0.8, 0.5]
```

进行一维卷积时，卷积核会滑动到数据的每个位置，然后在每个位置上执行元素乘法并相加。例如，对于位置1，卷积操作为：

```python
result[1] = 20 * 0.5 + 22 * 0.8 + 24 * 0.5 = 37.0
```

同样地，对于位置2，卷积操作为：

```python
result[2] = 22 * 0.5 + 24 * 0.8 + 25 * 0.5 = 47.0
```

继续这个过程，直到对整个数据进行卷积操作，得到平滑后的结果：

```python
smoothed_data = [37.0, 47.0, 56.5, 59.0, 63.0, 61.5, 51.5]
```

在这个例子中，我们使用一维卷积核来平滑气温数据，从而减少数据中的噪声，更好地观察气温的整体变化趋势。

##### 原理概述

一维卷积是指**在单个方向（通常是时间轴）上进行的卷积操作**。通常用于**序列模型、自然语言处理**领域该层创建卷积的卷积核输入层在单个空间（或时间）维度上以产生输出张量。如果“use_bias”为True，则**创建一个偏置向量并将其添加到输出**中。最后如果“激活”不是“无”，它也应用于输出。当使用该层作为模型中的第一层时，提供“input_shape”参数（整数元组或“无”，例如。`对于128维向量的10个向量的序列，或对于128维向量的可变长度序列为“（None，128）”。

一维卷积操作的原理与二维卷积类似，都是通过**滑动一个固定大小的卷积核（即滤波器）在输入上进行卷积操作**。在一维卷积中，卷积核是一个长度为 `k` 的一维张量，用于对输入的每个时间步进行滤波操作。卷积核的大小会影响到卷积后的输出形状，具体可以使用下面的公式计算：

$\text{output-length} = \lfloor \frac{\text{input-length} - \text{kernel-size} + 2\text{padding}}{\text{stride}} + 1 \rfloor$

其中，`input_length` 是输入张量的时间步数，`kernel_size` 是卷积核的大小，`padding` 是补零操作的大小，`stride` 是卷积核在输入上滑动的步幅。

![img](core algorithm.assets/20200605102836301.png)

`layers.Conv1D` 层可以设置多个参数，例如卷积核的**大小、步幅、填充方式、激活函数**等等。通过调整这些参数，可以有效地**提取输入数据中的时序特征**，用于后续的分类、回归等任务。

假设输入的数据为 $x$，卷积核为 $w$，偏置为 $b$，步长为 $s$，padding的大小为 $p$。

**对于一维卷积，我们可以将 $x$ 和 $w$ 的维度都表示为长度**，即：

$x=[x_1,x_2,x_3,…,x_n]$

$w=[w_1,w_2,w_3,…,w_m]$

则在不考虑padding的情况下，输出的每个元素 $y_i$ 可以表示为：

![image-20230326095119288](core algorithm.assets/image-20230326095119288.png)

其中，$i$ 表示输出的位置，$j$ 表示卷积核的位置，$s$ 表示步长。而考虑padding的情况下，可以将 $x$ 在两端分别加上 $p$ 个 0，然后代入上述公式即可。 

需要注意的是，一般情况下我们会在卷积层后面**添加一个激活函数来引入非线性**。在这个公式中，我们没有考虑激活函数的影响。

![在这里插入图片描述](core algorithm.assets/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiN6LSf6Z-25Y2O4YOm,size_20,color_FFFFFF,t_70,g_se,x_16.png)

卷积过程如上图所示，输入向量的大小为20，卷积核大小为5，步长（每一步移动距离）为1，不考虑填充，那么输出向量的大小为(20 - 5) / 1 + 1 = 16；如果考虑填充，那么输出向量大小为20 / 1 = 20。

更一般的，假设输入向量大小为F，卷积核大小为K， 步长为S，填充方式为“VALID”（也就是不考虑填充），那么输出向量大小**N= (F - K / S) + 1**；如果填充方式为“SAME”（不考虑步长，使输入矩阵和输出矩阵大小一样），则输出向量大小**N = F / S**

##### 参数详解

```python
tf.keras.layers.Conv1D(filters, 
					   kernel_size, 
					   strides=1, 
					   padding='valid',
					   data_format='channels_last', 
					   dilation_rate=1, 
					   groups=1,
					   activation=None, 
					   use_bias=True, 
					   kernel_initializer='glorot_uniform',
					   bias_initializer='zeros',
					   kernel_regularizer=None,
					   bias_regularizer=None, 
					   activity_regularizer=None, 
					   kernel_constraint=None,
					   bias_constraint=None, 
					   **kwargs
)
```

- `filters`: 整数，输出空间的维度（即卷积核的个数）。
- `kernel_size`: 整数或由一个整数构成的元组/列表，卷积核的空间或时间维度大小。
- `strides`: 整数或由一个整数构成的元组/列表，卷积核的步长。默认为 1。
- `padding`: 字符串，补齐策略（'valid' 或 'same'）。默认为 'valid'。
- `activation`: 字符串或可调用对象，激活函数。如果不指定，将不应用任何激活函数。
- `use_bias`: 布尔值，是否使用偏置。
- `kernel_initializer`: 卷积核的初始化器。如果不指定，将使用默认的 `Glorot `均匀分布初始化。
- `bias_initializer`: 偏置的初始化器。如果不指定，将使用默认的零初始化。
- `kernel_regularizer`: 卷积核的正则化器，可以使用 L1、L2 等正则化方式。
- `bias_regularizer`: 偏置的正则化器，可以使用 L1、L2 等正则化方式。
- `activity_regularizer`: 输出的正则化器，可以使用 L1、L2 等正则化方式。
- `kernel_constraint`: 卷积核的约束，可以使用非负值约束、最大范数约束等。
- `bias_constraint`: 偏置的约束，可以使用非负值约束、最大范数约束等。

1D卷积层（例如时间卷积）。通常用于**序列模型、自然语言处理**领域该层创建卷积的卷积核输入层在单个空间（或时间）维度上以产生输出张量。如果“use_bias”为True，则**创建一个偏置向量并将其添加到输出**中。最后如果“激活”不是“无”，它也应用于输出。当使用该层作为模型中的第一层时，提供“input_shape”参数（整数元组或“无”，例如。`对于128维向量的10个向量的序列，或对于128维向量的可变长度序列为“（None，128）”。

https://blog.csdn.net/weixin_49346755/article/details/124267879

```python
# 一维卷积层，输出形状为(None, 16, 8)，定义input_shape作为第一层
tf.keras.layers.Conv1D(8, 5, activation="relu", input_shape=(20, 1))
"""
 filters: 过滤器：整数，输出空间的维度（即卷积中输出滤波器的数量）
 kernel_size: 单个整数的整数或元组/列表，指定1D卷积窗口的长度。
 activation: 要使用的激活功能。激活函数
 strides: 步长，默认为1.
 padding: 表示填充方式，默认为VALID，也就是不填充。
 kernel_initializer: “内核”权重矩阵的初始化器(参见“keras.initizers”）。
 use_bias: 表示是否使用偏置矩阵，默认为True
 bias_initializer: 表示使用的偏置矩阵。
 
 Input shape:
    3D tensor with shape: `(batch_size, steps, input_dim)`

"""
# regularizers.l2(0.01) L2正则化(L2正则化因子)。
Con1 = layers.Conv1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01))(BN)
```

##### 输入与输出

需要注意的是，**该层的输入应该是一个三维张量，形状为 `(batch_size, steps, input_dim)`**，其中 `steps` 表示时间步数，`input_dim` 表示每个时间步的输入特征维度。该层的输出是一个三维张量，形状为 `(batch_size, new_steps, filters)`，其中 `new_steps` 是经过卷积和补齐后的时间步数，与原来的时间步数有关。

卷积层的输出也是一个张量，其**形状取决于卷积层的参数设置**。在一维卷积层中，如果使用padding="valid"，那么输出的形状为(batch_size, output_length, filters)，其中output_length表示输出序列的长度，filters表示卷积核的个数（**即输出序列每个位置上的特征维度数量**）。如果使用padding="same"，那么输出的形状为(batch_size, input_length, filters)，即与输入序列的长度保持一致。

需要注意的是，在卷积层中，**每个卷积核的参数对于输入是共享的**，即卷积核在输入张量的每个位置上进行卷积时使用的是相同的参数。这样可以大大减少模型的参数数量，同时也可以增强模型的泛化能力。

> 如果使用多个卷积核进行卷积操作，它们所提取的特征可能不同，因为它们所学习的卷积核参数不同。每个卷积核学习到的是不同的特征，通过**使用多个卷积核，模型可以同时学习到多种不同的特征**，从而提高模型的性能。

 ~~输入是二维~~

> ~~举例子如: 第一次卷积层后的维度是（None, 431,64) 输入到同样的卷积层（卷积核个数64个，长度3）输出维度也是（None, 429,64)~~

~~每个卷积核卷积时是在所有64列上进行的，因此每个卷积核的输出将是一个1维向量。**64 个卷积核的输出将组合成一个 64 一维向量，即使进行多次卷积，也不会导致输出的维度增加**。在卷积层中使用多个卷积核的原因是为了捕获输入数据中的不同特征。每个卷积核的输出是一个捕获到的特征映射，这些特征映射通过叠加的方式组成了一个64维向量。具体地说，**这个64维向量是将每个特征映射在时间轴上的输出值取平均得到的。这个过程可以看作是在对输入数据进行一种降维操作，将捕获到的不同特征映射组合成一个向量表示整个输入数据的特征**。~~

~~输入是三维~~



##### 多次卷积

在卷积层后再次添加卷积层是一种常见的神经网络架构，其主要目的是在学习更高层次的特征表示，例如在计算机视觉任务中，第一层卷积层可以学习到简单的边缘特征，而第二层卷积层可以学习到更加复杂的形状和纹理特征。因此，通过多层卷积层堆叠，可以逐渐学习到更加抽象和高级的特征表示。

##### 卷积核权重维度

如果你设置了六个长度为3的卷积核，那么每个卷积核的**权重矩阵**的形状将会是`(3, input_channels, 6)`，其中`input_channels`是输入数据的**特征维度**。这表示每个卷积核都是一维的，其大小为3，且有6个不同的卷积核。在进行卷积运算时，输入数据中的**每个时刻都会和6个不同的卷积核进行卷积操作**，得到6个卷积后的输出结果，这些结果将被连接成一个更高维的输出张量。

假设我们有一个输入数据的维度为（6， 4， 3），表示有6个时间步，4个特征和3个通道。我们想要应用一个大小为（3， 3）的卷积核。（卷积核的权重维度将是（3， 3， 3， 1））

#### Conv2D



就同我们在CNN章节开头所说看电影的生活例子，就是一个二维卷积的例子

一维卷积和二维卷积的区别在于卷积操作的维度不同。在一维卷积中，卷积核只会在一个方向上进行滑动操作，例如在处理时间序列数据时，卷积核只会在时间轴上进行滑动操作。而在二维卷积中，卷积核会在两个方向上进行滑动操作，例如在处理图像数据时，卷积核会在图像的高度和宽度上进行滑动操作。因此，一维卷积和二维卷积的计算方式略有不同，但**本质上都是将卷积核与输入数据进行点积运算，得到特征图作为下一层的输入。**

![img](core algorithm.assets/v2-cdbe7b7a84e9d34e954fac922e9404ab_b.jpg)

如上图所示，输入矩阵的大小为5×5，卷积核矩阵的大小为3×3，在x, y 方向移动步长为(1, 1)，采用了填充的方式（SAME）进行卷积（填充不是结果填充，是原本的填充。结果得到一个与输入矩阵大小一样的矩阵（5×5）。

![img](core algorithm.assets/v2-705305fee5a050575544c64067405fce_b.jpg)

卷积：蓝色的输入图片（4 x4）,深蓝色代表卷积核（3 x 3）,绿色为输出图像（2 x 2）

二维卷积的计算公式与一维卷积的计算公式类似，假设输入图像的大小为F×F，卷积核矩阵大小为K×K，步长为（S，S），如果填充方式为VALID，输出图像大小为N×N，则有**N = (F - K / S) + 1**；如果填充方式为SAME，则有**N = F / S**。

> 在神经网络中，**点积经常用于计算相似度、相似性分数或计算注意力权重等任务**。点积运算是指**两个向量中对应位置的元素相乘，并将所有结果相加的运算**。对于两个长度为n的向量a和b，它们的点积运算结果为：
>
> $a·b = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]$
>
> **两个向量的点积可以表示它们的相似度**，从而用于计算神经元的输出值或者用于计算损失函数。另外，在计算卷积神经网络中的卷积操作时，通常采用卷积核和输入数据的点积运算来得到卷积的结果。	
>
> > 点积本身并不能直接表示相似度，而是作为相似度度量的一种计算方式之一。**当两个向量的点积较大时，表示它们在相同的方向上有更高的相似度。而当点积较小或为负数时，表示它们在相反的方向上或无关的方向上存在较高的差异。** 通过点积，我们可以得到一种衡量向量之间关系的指标，但具体的解释和应用取决于具体的上下文和任务。

在二维卷积层中，输出的形状也取决于卷积层的参数设置，但是其基本形式为(batch_size, output_height, output_width, filters)，其中output_height和output_width表示输出特征图的高度和宽度，filters表示卷积核的个数（即输出特征图每个位置上的特征维度数量）。

```python
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

#### Conv3D

在三维卷积中有两种例子，其中之一假设你有一张**彩色图像，其中包含红、绿、蓝三个颜色通道。这样的彩色图像可以表示为一个三维数组，其中每个元素包含了图像在特定位置的颜色信息。**假设你想要对图像应用一个卷积核来进行边缘检测。边缘检测可以帮助我们找到图像中的边界或轮廓。这时，我们需要使用一个三维卷积核来处理彩色图像的每个颜色通道。考虑一个简单的三维卷积核，形状为 3x3x3，表示在3个颜色通道上的3x3的局部感知区域。卷积核会在图像的每个位置滑动，并执行元素乘法和相加操作，以获取特定位置的输出值。例如，对于图像中的某个位置，卷积操作会在每个颜色通道上执行元素乘法和相加，得到一个输出值。这个操作会在图像的所有位置重复进行，从而生成一个新的三维输出图像。这个例子中的三维卷积核用于边缘检测时，会对图像的每个颜色通道执行类似于边缘检测的操作。**它可以帮助我们在每个颜色通道上找到图像中的边缘或轮廓。**

还有一个例子是视频行为识别。假设你正在监控一间会议室，里面有多个人在进行不同的活动，例如站立、走动、举手等。你想要使用卷积神经网络来识别不同的行为。在这个场景中，视频可以看作是一个三维数据，它**由两个空间维度（图像的宽度和高度）和一个时间维度（视频的帧数）组成**。这样的视频可以表示为一个三维数组，其中每个元素代表一个像素值或颜色信息。为了对视频进行行为识别，我们需要使用三维卷积核来处理视频数据。这个卷积核在空间维度上滑动，同时在时间维度上遍历视频的帧，执行元素乘法和相加操作，以获取特定位置和时间的输出值。例如，考虑一个形状为 3x3x3x3 的三维卷积核，其中前两个维度表示在一个3x3的局部感知区域内，每个颜色通道的像素值，最后一个维度表示卷积核在时间维度上遍历视频的3帧。在应用三维卷积时，卷积核会在视频的每个位置和每个时间点滑动，并对每个颜色通道执行元素乘法和相加操作，得到一个输出值。这样的操作会在整个视频上重复进行，生成一个新的三维输出，**表示不同时间点和空间位置的特征**。这个例子中的三维卷积核用于视频行为识别时，可以帮助我们捕捉不同行为在时间序列上的特征变化。例如，当某人举手时，可能在一段时间内会出现特定的**手臂移动模式（一种数据变化模式）**，而这个三维卷积可以帮助我们捕捉这种时间序列上的模式。

三维卷积层主要用于医学领域、视频处理领域（检测人物行为），用于三个维度的卷积。

![在这里插入图片描述](../../../../../../Python全栈开发/人工智能/人工智能笔记本&案例/机器学习笔记本&案例/深度学习笔记本&案例/通用学习笔记/Keras/layers.assets/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiN6LSf6Z-25Y2O4YOm,size_12,color_FFFFFF,t_70,g_se,x_16%23pic_left.png)

三维卷积对数据集应用三维过滤器，过滤器向3个方向（x，y，z）移动，计算低层特征表示。输出形状是一个三维体积空间，如立方体或长方体。有助于视频、三维医学图像等的目标物检测。

三维卷积的输入形状为五维张量(batch_size, frames, height, width, channels)，batch_size为批处理数据的大小，frams可以理解为视频中的帧数，其中每一帧为一幅图像，height为图像的高度，width为图像的宽度，channels为图像通道数。输出形状也是一个五维张量。

#### 反卷积

> 假设你有一张古老的照片，由于年代久远和物理损坏，照片上出现了许多破损、划痕和噪点。你希望使用反卷积来修复照片，恢复其原始的清晰度和细节。在这个场景中，反卷积可以用于学习照片修复过程，将破损和损坏的像素映射到原始清晰的像素。这样的修复过程可以通过训练反卷积网络来实现。反卷积网络使用反卷积层来进行图像修复。**输入是破损的图像，输出是修复后的图像**。例如，考虑一个简单的反卷积层，输入是一张破损的图像（例如，256x256像素），输出是一张修复后的图像（例如，256x256像素）。在反卷积过程中，网络会使用反卷积核来将破损和损坏的像素恢复到原始图像的空间维度上。通过**训练网络学习图像的修复过程**，它可以从破损的输入图像中恢复出丢失的细节和结构，使修复后的图像更加清晰和自然。
>
> 实际应用中，反卷积在图像修复和复原中起着重要作用。它可以帮助修复老旧照片、受损图像或受到噪声污染的图像，恢复它们的原始外观和细节。这个生活例子展示了反卷积在图像修复中的应用，通过学习将破损的像素映射到原始清晰的像素，实现图像的修复和恢复。

反卷积（deconvolution）也称为转置卷积（transpose convolution），是一种常用于图像处理和计算机视觉领域的操作。它可以将一个低维度的特征映射（如经过卷积和池化后的结果）还原成更高维度的特征映射，通常用于图像分割、目标检测等任务中。

它是一种特殊的卷积，先padding来扩大图像尺寸，紧接着跟正向卷积一样，旋转卷积核180度，再进行卷积计算。看上去就像，已知正向卷积的输出图像，卷积核，得到正向卷积中的原始图像（并非真的得到原始图像，**像素点是不一样的，但是尺寸是一致的**）。它看上去像是正向卷积的逆运算，但其实并不是。因为反卷积只能还原原始图像的尺寸，但是并不能真的恢复原始图像内容，即每个元素值其实是不一样的。

> 由于卷积核一般比原始图像小，所以卷积之后的图像尺寸往往会变小。有时候我们需要将卷积后的图像还原成原始图像的尺寸，即实现图像从小分辨率到大分辨率的映射，这种操作就叫做上采样（Upsampling）。而反卷积正是一种上采样方法。

(参考自： https://www.cnblogs.com/hansjorn/p/14767592.html)

反卷积的Striding跟卷积有点不一样，它在输入的每个元素之间插入 $s- 1$个值为0的元素

![img](core algorithm.assets/v2-cd812a71025fd95e0a1ced13d6df8155_720w.jpg)

我们根据以下例子来了解原理：

![img](core algorithm.assets/v2-4992be8ec58775d0f6f963c2ae7129b3_b.jpg)

反卷积：蓝色是输入（3 x 3）, 灰色是卷积核（3 x 3）, 绿色是输出（5 x 5），padding=1，strides = 2

![img](core algorithm.assets/v2-cdbe7b7a84e9d34e954fac922e9404ab_b-16878543936166.jpg)

反卷积：蓝色是输入（5 x 5）, 灰色是卷积核（3 x 3）, 绿色是输出（5 x 5），padding=1，strides =1

![img](core algorithm.assets/v2-286ac2cfb69abf4d8aa06b8eeb39ced3_b.jpg)

反卷积，蓝色是输入（2 x 2）, 灰色是卷积核（3 x 3）, 绿色是输出（4 x 4），padding=2

应用场景：

1. 图像超分辨率：反卷积可以将低分辨率的图像还原为高分辨率，这在图像超分辨率任务中非常有用。
2. 图像去噪：反卷积可以去除图像中的噪声，从而改善图像质量。
3. 图像分割：反卷积可以将卷积网络的最后一层输出特征图还原成与输入图像尺寸相同的特征图，从而帮助进行图像分割。
4. 对抗生成网络（GAN）：反卷积可以用于生成器部分，将低维噪声转换为高维图像。
5. 图像生成：逆卷积操作可以通过学习合适的卷积核来生成新的图像，从而实现图像生成任务。

总的来说，反卷积在计算机视觉领域中有着广泛的应用场景。它是一种非常有效的图像处理技术，可以帮助我们提高计算机视觉任务的准确性和效率。

#### 空间可分离卷积

空间可分离卷积（Spatial Separable Convolution）算法是一种用于图像处理和计算机视觉任务的卷积操作优化技术。它的主要特点是将二维卷积操作拆分成两个一维卷积操作，从而显著降低计算复杂度。

空间可分离卷积的历史发展可以追溯到上世纪80年代。这个想法最初由Grossmann和Morlet于1984年提出，用于处理信号上的小波变换。后来，它被引入到图像处理和计算机视觉领域，成为一种常用的卷积操作优化方法。

空间可分离卷积的灵感源自于观察到，二维卷积操作可以分解为两个独立的一维卷积操作。具体来说，对于一个M×N的输入图像和一个K×K的卷积核，传统的二维卷积操作需要执行M×N×K×K次乘法运算。而空间可分离卷积算法将其分解为一个M×K的水平卷积操作和一个K×N的垂直卷积操作，分别执行M×K×K和K×N×K次乘法运算。这样，总的乘法运算次数减少为M×K×K + K×N×K，大大降低了计算复杂度。

除此之外，该算法可以对多通道序列很好学习到**时域和特征域**。

空间可分离卷积的主要特点和结构如下：
1. 分解性质：空间可分离卷积将二维卷积操作分解为两个独立的一维卷积操作。这种分解性质使得卷积操作的计算复杂度降低，同时也方便了算法的实现和优化。
2. 计算效率：由于分解后的一维卷积操作具有更低的计算复杂度，空间可分离卷积算法在图像处理和计算机视觉任务中能够显著加快计算速度，提高算法的实时性。
3. 空间相关性：空间可分离卷积算法充分利用了图像中的空间相关性。一维卷积操作分别在水平和垂直方向上进行，更好地捕捉了图像中的水平和垂直特征，提高了算法的表达能力。
4. 可扩展性：空间可分离卷积算法可以与其他卷积算法结合使用，如深度卷积神经网络（CNN）。它可以作为CNN中的一种基础卷积操作，用于提取图像特征。

关于学习资源，以下是一些相关的参考资料和论文：
1. Gonzalez, R. C., & Woods, R. E. (2008). Digital Image Processing (3rd ed.). Prentice Hall.
2. Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444.
4. Grossmann, A., & Morlet, J. (1984). Decomposition of Hardy Functions into Square Integrable Wavelets of Constant Shape. SIAM Journal on Mathematical Analysis, 15(4), 723–736.
5. Mallat, S. (1989). A Theory for Multiresolution Signal Decomposition: The Wavelet Representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(7), 674–693.

#### 预训练卷积神经网络（最佳实践）

代替循环神经网络

在面对没有时间序列结构信息，CNN会更加好用，且更快速



循环神经网络是将信息存在了隐藏层往后传递，由于这个原因，在处理时序数据中是一个时间步一个时间步的输入，在一百个时间步的情况下，需要先算出前99个时间步才能算出第100个，所以并行能力较差，在计算机性能表示方面较差。同样还是这个原因，在序列比较长的话，长短期记忆较差。在并行上也就有人提出了使用CNN来代替RNN。

### 循环神经网络

在卷积神经网络和MLP全连接神经网络中，数据点在神经层都是互相独立无联系的，在输入与输入之间没有保存任何状态。要想处理数据点的序列或时间序列，你需要**向网络同时展示整个序列，即将序列转换成单个数据点**。

> 与此相反，当你在阅读这个句子时，你是一个词一个词地阅读（或者说，眼睛一次扫视一次扫视地阅读），同时会记住之前的内容。这让你能够动态理解这个句子所传达的含义。生物智能以渐进的方式处理信息，同时保存一个关于所处理内容的内部模型，这个模型是根据过去的信息构建的，并随着新信息的进入而不断更新。

循环神经网络（RNN，recurrent neural network）采用同样的原理，**在形式上也是一个序列整体输入，但不同的是在内部中参数对每个时刻是共享的，每个时刻的数据步依次进行输入，在新的序列中状态被重置**。` timedistribute(dense) `(时间序列输入的全连接层）也是同样的思想。

下面是RNN、LSTM和GRU三者的优缺点的比较：

| 模型 | 优点                                                         | 缺点                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RNN  | - 简单的结构和参数<br>- 可以处理序列数据的时间依赖关系<br>- 计算效率较高 | - **难以处理长期依赖关系**<br>- 容易遇到**梯度消失**或**梯度爆炸**问题 |
| LSTM | - 能够**捕捉和处理长期依赖关系**<br>- 通过门控机制控制信息流动<br>- 网络结构相对简单 | - 参数较多，**计算开销较大**<br>- 可能会出现过拟合问题       |
| GRU  | - 相对于LSTM，**参数更少，计算效率更高**<br>- 通过门控机制控制信息流动<br>- 可以捕捉长期依赖关系 | - 可能会失去一些细粒度的时间信息<br>- 对于某些复杂的序列问题，性能可能不如LSTM |

实际应用中，选择合适的模型取决于具体任务和数据集的特点。有时候，LSTM可能更适合捕捉长期依赖关系，而GRU则具有更高的计算效率，这是在深度学习经常遇到的问题，在性能表现得计算效率中的取舍。

![img](core algorithm.assets/1B0q2ZLsUUw31eEImeVf3PQ.png)

详细讲解RNN,LSTM,GRU  https://towardsdatascience.com/a-brief-introduction-to-recurrent-neural-networks-638f64a61ff4

#### RNN （Recurrent Neural Network）

循环神经网络（Recurrent Neural Network，RNN）是一种用于处理序列数据的神经网络算法。相比于传统的前馈神经网络，RNN引入了**循环连接**，使网络能够对序列中的**时间依赖关系**进行建模。

>  RNN的核心思想是在网络的隐藏层之间引入循环连接，使得网络在处理每个时间步的输入时，不仅依赖当前时间步的输入，还依赖前一个时间步的隐藏状态。这种循环连接的设计使得网络具有记忆性，能够捕捉到序列数据中的长期依赖关系。

下面是RNN的算法步骤：

![img](core algorithm.assets/1iP_ahgzkiMNu2hPYhkjlXw.png)

公式如下:

<img src="core algorithm.assets/10gFfVWuKCjYaYE5_oFy9EQ@2x.png" alt="img" style="zoom:50%;" />

1. 初始化参数：
   - 定义输入序列的维度（例如，词向量的维度）和隐藏状态的维度。
   - 随机初始化权重矩阵，如输入到隐藏层的权重矩阵、隐藏层到隐藏层的权重矩阵和隐藏层到输出层的权重矩阵。
2. 前向传播：
   - 对于输入序列中的每个时间步，将输入向量和前一个时间步的隐藏状态向量输入到隐藏层中。
   - 根据当前时间步的输入和前一个时间步的隐藏状态，计算隐藏层的输出。
   - 使用隐藏层的输出计算当前时间步的预测结果（可以是分类问题的概率分布或回归问题的连续值）。
3. 更新隐藏状态：
   - 使用当前时间步的隐藏状态更新到下一个时间步的隐藏状态。通常使用tanh或ReLU等激活函数来处理隐藏状态。
4. 计算损失：
   - 根据预测结果和真实标签计算损失函数（如交叉熵损失或均方误差）。
5. 反向传播：
   - 计算损失函数对于权重矩阵的梯度。
   - 使用梯度下降或其他优化算法更新权重矩阵。
6. 重复步骤2-5直到所有时间步的输入都被处理完毕，或达到指定的训练迭代次数。

RNN算法的核心是通过循环连接实现对序列数据的建模和预测。它在自然语言处理（NLP）领域广泛应用于语言模型、机器翻译、情感分析等任务，也适用于时间序列数据的预测和生成。

需要注意的是，传统的RNN在处理长期依赖关系时可能会遇到梯度消失或梯度爆炸的问题。为了克服这个问题，改进的RNN变体，如长短期记忆网络（LSTM）和门控循环单元（GRU），被提出并广泛应用。

#### LSTM（Long Short Time Memory)

##### 原理详解(每个神经元)

RNN 的最大问题是，理论上来说，在时刻 t 应该能够记住许多时间步之前见过的信息，但实际上它是不可能学到这种长期依赖的。其原因在于梯度消失问题（vanishing gradient problem），这一效应类似于在层数较多的非循环网络（即前馈网络）中观察到的效应：随着层数的增加，网络最终变得无法训练。Hochreiter、Schmidhuber 和 Bengio 在 20 世纪 90 年代初研究了这一效应的理论原因。LSTM 层和 GRU 层都是为了解决这个问题而设计的。

> LSTM 层其背后的长短期记忆（LSTM，long short-term memory）算法由 Hochreiter和 Schmidhuber 在 1997 年开发 ，是二人研究梯度消失问题的重要成果。

LSTM 层作为RNN 层的一种变体，它增加了一种携带信息跨越多个时间步的方法。假设有一条传送带，其运行方向**`平行于你所处理的序列`**。序列中的信息可以在任意位置跳上传送带，然后被传送到更晚的时间步，并在需要时原封不动地跳回来。这实际上就是 LSTM 的原理：它**`保存信息以便后面使用，从而防止较早期的信号在处理过程中逐渐消失`**。

> 在实现上，LSTM层中有三个门控单元，即输入门、遗忘门和输出门。这些门控单元在每个时间步上控制着LSTM单元如何处理输入和记忆。具体来说，
>
> 在每个时间步中，输入$x_t$和前一时刻的隐状态$h_{t-1}$被馈送给门控制器，然后门控制器**根据当前的输入$x_t$和前一时刻的隐状态$h_{t-1}$（即上一时间步的输出）计算出三种门的权重**，然后将这些权重作用于前一时刻的记忆单元$c_{t-1}$。具体来说，门控制器计算出三个向量：**输入门的开启程度$i_t$、遗忘门的开启程度$f_t$和输出门的开启程度$o_t$，这三个向量的元素值均在[0,1]**之间。
>
> 通过使用这些门的权重对前一时刻的记忆单元$c_{t-1}$进行更新，**计算出当前时刻的记忆单元$c_t$**，并**将它和当前时刻的输入$x_t$作为LSTM的输出$y_t$**。最后，将**当前时刻的记忆单元$c_t$和隐状态$h_t$一起作为下一时刻的输入**，继续进行LSTM的计算。



![img](core algorithm.assets/1tEN1Ziu4VvRAaH9zagN3EQ.png)

LSTM的参数包括输入到状态的权重$W_{xi},W_{hi},b_i$，输入到遗忘门的权重$W_{xf},W_{hf},b_f$，输入到输出门的权重$W_{xo},W_{ho},b_o$，以及输入到记忆单元的权重$W_{xc},W_{hc},b_c$，其中$W$表示权重矩阵，$b$表示偏置向量。

###### a. 遗忘门：Forget Gate

遗忘门的功能是决定应丢弃或保留哪些信息。来自前一个隐藏状态的信息和当前输入的信息同时传递到 sigmoid 函数中去，输出值介于 0 和 1 之间，越接近 0 意味着越应该丢弃，越接近 1 意味着越应该保留。

![20210305193619275](core%20algorithm.assets/20210305193619275.gif)

遗忘门的计算公式

![在这里插入图片描述](core algorithm.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3OTI3NzMx,size_16,color_FFFFFF,t_70.png)

###### b. 输入门：Input Gate

输入门用于更新细胞状态。首先将前一层隐藏状态的信息和当前输入的信息传递到 sigmoid 函数中去。将值调整到 0~1 之间来决定要更新哪些信息。并**将前一层隐藏状态的信息和当前输入的信息传递到 tanh 函数中**去，创造一个新的侯选值向量。最后**将 sigmoid 的输出值与 tanh 的输出值相乘，sigmoid 的输出值将决定 tanh 的输出值中哪些信息是重要且需要保留下来**的

> 使用tanh作为LSTM输入层的激活函数是比较常见的做法。在LSTM中，如果权重值较大或者较小，那么在反向传播时，梯度值会非常大或者非常小，导致梯度爆炸或者消失的情况。而**tanh函数的导数范围在[-1, 1]之间，可以抑制梯度的放大和缩小，从而避免了梯度爆炸和消失的问题(RNN遇到的问题）**。此外，tanh函数在输入为0附近的时候输出接近于线性，使得网络更容易学习到线性相关的特征。另外，tanh 函数具有对称性，在处理序列数据时能够更好地捕捉序列中的长期依赖关系。
>

![20210305193712444](core%20algorithm.assets/20210305193712444.gif)

输入门的计算公式

![在这里插入图片描述](core algorithm.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3OTI3NzMx,size_16,color_FFFFFF,t_70-16801340054184.png)

###### c. Cell State 传播

首先前一层的细胞状态与遗忘向量逐点相乘。如果它乘以接近 0 的值，意味着在新的细胞状态中，这些信息是需要丢弃掉的。然后再将该值与输入门的输出值逐点相加，将神经网络发现的新信息更新到细胞状态中去。至此，就得到了更新后的细胞状态。

![20210305193746597](core%20algorithm.assets/20210305193746597.gif)

Cell State的计算公式

![在这里插入图片描述](core algorithm.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3OTI3NzMx,size_16,color_FFFFFF,t_70-16801340172377.png)

###### d. 输出门：Output Gate

输出门用来确定下一个隐藏状态的值，**隐藏状态包含了先前输入的信息**。首先，我们将前一个隐藏状态和当前输入传递到 sigmoid 函数中，然后将新得到的细胞状态传递给 tanh 函数。最后将 tanh 的输出与 sigmoid 的输出相乘，以**确定隐藏状态应携带的信息**。再将隐藏状态作为当前细胞的输出，把新的细胞状态和新的隐藏状态传递到下一个时间步长中去。

![20210305193815841](core%20algorithm.assets/20210305193815841.gif)

输出门的计算公式

![在这里插入图片描述](core algorithm.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3OTI3NzMx,size_16,color_FFFFFF,t_70-168013403191610.png)



在以上的讲解中，我讲的哲学一点，解释每个运算的目的。比如遗忘门中的运算是为了故意遗忘携带数据流中的不相关信息。同时，输入门用新信息来更新携带轨道。但归根结底，这些解释并没有多大意义，因为这些运算的实际效果

是由**参数化权重**决定的，而权重是以端到端的方式进行学习，每次训练都要从头开始，**不可能为某个运算赋予特定的目的**。RNN 单元的类型（如前所述）决定了你的假设空间，即在训练期间搜索良好模型配置的空间，但它不能决定 RNN 单元的作用，那是由单元权重来决定的。同一个单元具有不同的权重，可以实现完全不同的作用。因此，组成 RNN 单元的运算组合，最好被解释为**对搜索的一组约束，而不是一种工程意义上的设计**。

对于研究人员来说，这种约束的选择（即如何实现 RNN 单元）似乎最好是留给最优化算法来完成（比如遗传算法或强化学习过程），而不是让人类工程师来完成。在未来，那将是我们构建网络的方式。总之，你不需要理解关于 LSTM 单元具体架构的任何内容。作为人类，理解它不应该是你要做的。你**`只需要记住 LSTM 单元的作用`**：允许过去的信息稍后重新进入，从而**解决梯度消失问题。**

#####  模型参数计算

对于一个LSTM（长短期记忆）模型，参数的计算涉及输入维度、隐藏神经元数量和输出维度。在给定输入维度为（64，32）和LSTM神经元数量为32的情况下，我们可以计算出以下参数：

1. 输入维度：（64，32）
   这表示每个时间步长（sequence step）的输入特征维度为32，序列长度为64。

2. 隐藏神经元数量：32
   这是指LSTM层中的隐藏神经元数量。每个时间步长都有32个隐藏神经元。

3. 输入门参数：
   - 权重矩阵：形状为（32，32 + 32）的矩阵。其中32是上一时间步的隐藏状态大小，另外32是当前时间步的输入维度。
   - 偏置向量：形状为（32，）的向量。

4. 遗忘门参数：
   - 权重矩阵：形状为（32，32 + 32）的矩阵。
   - 偏置向量：形状为（32，）的向量。

5. 输出门参数：
   - 权重矩阵：形状为（32，32 + 32）的矩阵。
   - 偏置向量：形状为（32，）的向量。

6. 单元状态参数：
   - 权重矩阵：形状为（32，32 + 32）的矩阵。
   - 偏置向量：形状为（32，）的向量。

7. 输出参数：
   - 权重矩阵：形状为（32，32）的矩阵。将隐藏状态映射到最终的输出维度。
   - 偏置向量：形状为（32，）的向量。

因此，总共的参数数量可以通过计算上述所有矩阵和向量中的元素总数来确定。



在文本中由于其句子正向逆向角度来阅读信息其实都差不多，但是能够补充一点信息，而在正向时序强的话效果就相对没有那么好了

**案例实现代码：** 下面是一个使用Keras库实现双向LSTM进行情感分析的示例代码：

```PYTHON
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense

# 定义模型
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(units=hidden_units)))
model.add(Dense(units=output_classes, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
```

以上代码用到了Keras库来构建双向LSTM模型。首先，定义了一个Sequential模型，并添加了一个嵌入层（Embedding）作为输入层。接下来，通过Bidirectional函数将LSTM层包裹起来，形成双向LSTM结构。最后，添加了一个全连接层（Dense）作为输出层，使用softmax激活函数进行分类。（这里除了concatenate 还可以summing)

>  In a bidirectional LSTM, the default method for combining the forward and backward hidden states is concatenation, not summing. The concatenated hidden states provide a more comprehensive representation that retains both the forward and backward information separately in different dimensions.
>
>  However, if you specifically want to use summing instead of concatenation to combine the forward and backward hidden states, you would need to modify the implementation of the bidirectional LSTM layer.
>
>  Here's an example of how you can modify the bidirectional LSTM implementation to use summing instead of concatenation:
>
>  ```python
>  import tensorflow as tf
>  
>  class SummingBidirectionalLSTM(tf.keras.layers.Layer):
>    def __init__(self, hidden_units):
>        super(SummingBidirectionalLSTM, self).__init__()
>        self.hidden_units = hidden_units
>        self.forward_lstm = tf.keras.layers.LSTM(units=hidden_units, return_sequences=True)
>        self.backward_lstm = tf.keras.layers.LSTM(units=hidden_units, return_sequences=True, go_backwards=True)
>  
>    def call(self, inputs):
>        forward_output = self.forward_lstm(inputs)
>        backward_output = self.backward_lstm(inputs)
>        summed_output = forward_output + backward_output
>        return summed_output
>  ```
>
>  In this modified implementation, the forward and backward LSTM layers are created separately, and their outputs are summed element-wise using the `+` operator. The `summed_output` represents the combined hidden states obtained by summing the forward and backward information.
>
>  Please note that using summing instead of concatenation may result in a different representation and may affect the model's performance on specific tasks. It's recommended to carefully evaluate the impact of this modification on your specific use case before adopting it.

然后，通过编译模型，并使用训练数据进行模型训练。在训练完成后，可以使用测试数据评估模型的性能，计算损失和准确率（accuracy）。请注意，上述代码仅为示例，实际使用时需要根据具体任务进行适当修改和调整。

如果你对LSTM以及其与反向传播算法之间的详细联系感兴趣，我建议你参考以下资源：

1. BENGIO Y, SIMARD P, FRASCONI P. Learning long-term dependencies with gradient descent is difficult [C]//IEEE Transactions on Neural Networks, 1994, 5(2): 157-166.
2. HOCHREITER S, SCHMIDHUBER J. Long short-term memory [J]. Neural Computation, 1997, 9(8): 1735-1780.
3. "Understanding LSTM Networks" by Christopher Olah: https://colah.github.io/posts/2015-08-Understanding-LSTMs/  强烈推荐！！！ 
4. TensorFlow官方教程：Sequence models and long-short term memory network (https://www.tensorflow.org/tutorials/text/text_classification_rnn)
5. PyTorch官方文档：nn.LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
6. 详细讲解RNN,LSTM,GRU  https://towardsdatascience.com/a-brief-introduction-to-recurrent-neural-networks-638f64a61ff4

#### GRU (Gated Recurrent Unit)

GRU（Gated Recurrent Unit，门控循环单元）是一种循环神经网络（RNN）的变体，用于处理序列数据。它是为了解决传统RNN中的梯度消失问题和长期依赖问题而提出的。

GRU单元通过引入门控机制来控制信息的流动，从而能够更好地捕捉长期依赖关系。相比于传统的RNN和LSTM，GRU具有更简单的结构和更少的参数，但仍能有效地建模和处理序列数据。

![img](core%20algorithm.assets/1-ldMy6GqBy8D25uNKQl2gA.png)

GRU单元的核心结构包括以下几个关键组件：

1. 更新门（Update Gate）：
   - 更新门决定了当前时间步的输入和前一个时间步的隐藏状态之间的权重。
   - 更新门的输出范围在0和1之间，用于控制信息流动的权重。接近1表示重要信息被保留，接近0表示信息被丢弃。

2. 重置门（Reset Gate）：
   - 重置门决定了前一个时间步的隐藏状态如何影响当前时间步的输入。
   - 重置门的输出范围在0和1之间，用于控制前一个时间步的隐藏状态对当前时间步的输入的影响程度。

3. 候选隐藏状态（Candidate Hidden State）：
   - 候选隐藏状态是基于当前时间步的输入和前一个时间步的隐藏状态计算得到的中间状态。
   - 它是一个候选的更新后的隐藏状态，用于在考虑前一个时间步的隐藏状态的同时，结合当前时间步的输入。

4. 最终隐藏状态：
   - 最终的隐藏状态是通过加权平均候选隐藏状态和前一个时间步的隐藏状态得到的。
   - 更新门的输出和候选隐藏状态决定了最终隐藏状态中保留的信息。

GRU单元的计算过程可以描述如下：

- 输入：当前时间步的输入（x_t）、前一个时间步的隐藏状态（h_{t-1}）
- 输出：当前时间步的隐藏状态（h_t）

1. 计算更新门（z_t）：
   - z_t = sigmoid(W_z * x_t + U_z * h_{t-1})

2. 计算重置门（r_t）：
   - r_t = sigmoid(W_r * x_t + U_r * h_{t-1})

3. 计算候选隐藏状态（h~_t）：
   - h~_t = tanh(W_h * x_t + r_t * U_h * h_{t-1})

4. 计算最终隐藏状态（h_t）：
   - h_t = (1 - z_t) * h_{t-1} + z_t * h~_t

GRU单元的参数包括权重矩阵（W和U）和偏置向量。这些参数通过反向传播算法进行训练，以最小化定义的损失函数。

相比于传统的RNN和LSTM，GRU单元的结构更简洁，具有更少的门控单元和参数。这使得GRU在处理序列数据时具有更高的计算效率，并能够更好地捕捉长期依赖关系。

#### 循环神经网络高级应用最佳实践

##### 合理分析问题

在面对一些情感分析问题时表现不好，主要原因在于，适用于**`评论分析全局的长期性结构`**（这正是 LSTM 所擅长的），对情感分析问题帮助不大。对于这样的基本问题，观察每条评论中出现了哪些词及其出现频率就可以很好地解决。**全连接方法的做法效果能够更好**。但还有更加困难的自然语言处理问题，特别是问答和机器翻译，这时 LSTM 的优势就明显了。

而需要注意的是，LSTM层在**处理长序列时容易出现梯度消失或爆炸的问题**。为了解决这个问题，通常会使用一些技巧，比如截断反向传播、梯度裁剪、残差连接等

##### 循环神经网络输出

以上讲解中最终输出是一个形状为 (timesteps, output_features) 的二维张量，其中每个时间步是循环在 t 时刻的输出。输出张量中的每个时间步 t 包含输入序列中时间步0~t 的信息，即关于全部过去的信息。因此，在多数情况下，你并不需要这个所有输出组成的序列，你只需要最后一个输出（循环结束时的 output_t），因为它已经**包含了整个序列的信息**。（如果需要使用LSTM的中间状态，可以在Keras将`return_sequences`参数设置为True）

##### 循环 dropout（recurrent dropout）

这是一种特殊的内置方法，在循环层中使用 dropout

来降低过拟合。

##### 堆叠循环层（stacking recurrent layers） & 双向循环层（bidirectional recurrent layer）

这会提高网络的表示能力（代价是更高的计算负荷）。

将相同的信息以不同的方式呈现给循环网络，可以提高精度并缓解遗忘问题。

##### 时间序列数据生成器

要注意的是提前进行数据预处理工作

编写一个 Python 生成器，以当前的浮点数数组作为输入，并从最近的数据中生成数据批量，同时生成未来的目标温度。因为数据集中的样本是高度冗余的（对于第 *N* 个样本和第 *N*+1 个样本，大部分时间步都是相同的），所以显式地保存每个样本是一种浪费。相反，我们将使用原始数据即时生成样本。

LSTM构建时间序列模型同样需要用到**滑动窗口**的概念构建数据集与标签。

<img src="core%20algorithm.assets/image-20231130110510200.png" alt="image-20231130110510200" style="zoom:50%;" />

以下是一个实现的基本代码，这个`lookback`其实我们也可以通过前面ARIMA模型求解的自相关与偏向关来参考，这里的适合小数据量

```python
def slip_windows(x, lookback=50):
    """
    :param x: Array-like data (one dimension)
    :param steps: slip step
    """
    assert len(x) > lookback, "参数应大于等于steps"

    data = []
    target = []
    for i in range(len(x) - lookback):
        data.append(x[i:i + lookback])
        target.append(x[i+lookback])
    return np.array(data), np.array(target)
```

实际上在面对大量数据时，一般建议使用生成器来加载数据，以下是一个适用于大部分常见的时间序列数据生成器代码，如果这里的数据较少反而会带来代码处理麻烦

```python
def generator(data, lookback, delay, min_index=0, max_index=None, shuffle=False, batch_size=128, step=6):
    """
    生成时间序列样本及其目标的生成器函数。

    参数：
    - data：包含时间序列数据的数组
    - lookback：输入数据的过去时间步数
    - delay：目标数据在未来的时间步数
    - min_index：数据的最小索引
    - max_index：数据的最大索引
    - shuffle：是否打乱样本顺序，默认为 False
    - batch_size：每个批次的样本数，默认为 128
    - step：数据采样的周期，默认为 6（每小时采样一个数据点）

    返回：
    - 生成器对象，每次迭代返回一个批次的样本和目标
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback  # 对于数据索引下标
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)  # samples indice
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback  # again new batchsize
            rows = np.arange(i, min(i + batch_size, max_index))  # samples indice
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay]
        yield samples, targets
           
        
        
# 测试代码
mean = np.mean(data_d[:130])
std = np.std(data_d[:130])
data_d -= mean
data_d /= std
train_gen = generator(data_d.values, max_index=100, lookback=lookback, delay=delay, step=step, batch_size=batch_size)
# 获取下一个批次的样本和目标
samples, targets = next(train_gen)

# 打印样本和目标的形状
print("样本形状:", samples.shape)
print("目标形状:", targets.shape)

# 打印样本和目标的内容
print("样本:")
print(samples)
print("目标:")
print(targets)
```



## Transformer家族

### Transformer base

论文《Attention is All you Need》

注意力（Attention）机制由Bengio团队与2014年提出并在近年广泛的应用在深度学习中的各个领域，例如在计算机视觉方向用于捕捉图像上的感受野，或者NLP中用于定位关键token或者特征。谷歌团队近期提出的用于生成词向量的BERT算法在NLP的11项任务中取得了效果的大幅提升，堪称2018年深度学习领域最振奋人心的消息。而BERT算法的最重要的部分便是本文中提出的Transformer的概念。

其中Transformer 模型架构图如下：

> Deocoder中的 K,V 为 Encoder 的输出（内容信息），Q 为 Decoder 中根据参考Token的查询信息

![img](core algorithm.assets/U3EwOx.png#shadow)

#### 背景和动机

在NLP模型领域中，seq2seq是一种常见的模型结构（序列到序列），其于 2013年、2014 年被多位学者共同提出，在机器翻译任务中取得了非常显著的效果，随后提出的 attention 模型更是将 Seq2Seq 推上了神坛，Seq2Seq+attention 的组合横扫了非常多的任务，只需要给定足够数量的 input-output pairs，通过设计两端的 sequence 模型和 attention 模型，就可以训练出一个不错的模型。除了应用在机器翻译任务中，其他很多的文本生成任务都可以基于 Seq2Seq 模型来做。

常见典型的任务有：机器翻译任务，文本摘要任务、代码补全、诗词生成等等，其思想不仅可以用在文本领域还可以用在语音图像领域中。

`那么在transfromers前传统的seq2seq任务实现方案是如何是实现的，有哪些缺点呢？`

传统的序列模型通常使用的是循环神经网络，如RNN（或者LSTM，GRU等），但是循环神经网络的计算限制为是顺序的，也就是说循环神经网络算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了三个问题：

1. 时间片 t 的计算依赖 t−1 时刻的计算结果，需要计算完前一个时间刻才能下一步计算这样限制了**模型的并行能力**；比如RNN中的t0时刻跟t10时刻的信息如果要交互，必须经过t1~t9，才能传递过去，信息且会随着传递距离增加而衰减，对信息的捕获能力较差，所求特征的表征能力也就更差了

2. 传统的序列模型存在着长期依赖问题，难以捕捉**长距离的依赖关系**。顺序计算的过程中**信息会丢失**，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象, LSTM依旧无能为力。而如果用CNN来代替RNN的解决方法（平行化)，但也只能**感受到部分的感受野**，**需要多层叠加**才能感受整个图像，其中可以参考下图辅助理解。

   <img src="core algorithm.assets/image-20230922114917294.png" alt="image-20230922114917294" style="zoom: 33%;" />
   
   

为了解决这个问题，作者提出一种新的注意力机制 *self attention* 结构，我们下面就看提出的这种结构如何解决上面的两个问题

 [**Self**-**attention** with relative position representations](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/1803.02155&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=5563767891081728261&ei=K57BZfHYArCH6rQPgome8AY&scisig=AFWwaeZGunl1eC-MkbFMxp5PVpEE)

#### Self Attention

Self Attention是Transformer模型的灵魂核心组件之一。该机制目的是让模型根据输入序列中**不同位置的相关性权重**来计算每个位置的表示，通过计算**查询和键之间的相似性**得分，并将这些得分应用于值来获取**加权和，从而生成每个位置的输出表示**。（其目的就是解决以上所说的两个问题）

> 这样我们在每个位置的序列输出都和全部位置的序列有关，这解决了第一个问题：全局的视野（对信息的捕获能力更强），同时该计算是各个向量矩阵点积运算，可以满足并行化运行，这就解决了第二个问题：时间片 t 的计算不依赖 t−1 时刻的计算结果，比如RNN中的t0时刻跟t10时刻的信息距离只是一个常量。

Self Attention接受的输入是三个相同的向量，分别命名为 Query 向量，一个 Key 向量和一个 Value 向量。

> 那我们面对输入序列X，如何满足Self Attention接受的输入呢？`

Q、K和V是通过**对输入序列进行线性变换得到**的，通过对输入序列的每个位置应用**不同的权重矩阵**，将输入序列映射到具有不同维度的查询（Q）、键（K）和值（V）空间。这样，我们就可以使用这些查询、键和值来输入到**Self Attention**结构计算注意力权重并生成加权表示。

<img src="core algorithm.assets/image-20230922120011708.png" alt="image-20230922120011708" style="zoom:33%;" />

给定一个输入序列X，我们可以通过线性变换得到Q、K和V：

Q = X * W_Q
K = X * W_K
V = X * W_V

其中W_Q、W_K和W_V是可学习的权重矩阵。

使用Q、K和V的好处是，它们允许模型根据**输入的不同部分**对相关信息进行加权。Q用于查询**输入序列的每个位置**，K用于提供关于**其他位置的信息**，V则提供用于计算加权表示的值。

值（V）代表的是确切的值（线性变换得到），一般是不变的用于求最后的输出，其次要实现求各个向量的相似性，如果只有一个k，而没有q，那k 与其他输入的 k作相似性，自己单元没有可以做相似性的地方，而再加一个q就可以实现了, 从而最后得到权重。

> 在第一次看到Q,K,V的时候我们会想，为什么需要这三个值呢？`

Self Attention 为了解决以上所说的两个问题，所采取的思路是通过全局位置的序列向量之间的相似性关系进行建模，来达到全局视野的目的，那么我们要计算每个位置向量之间的相似性权重，并指导当前位置的输出。

一种常用的计算相似度的方法是点积运算，而 Q,K 向量点击运算的结果每个位置向量之间（包括自己与自己）的相似性权重，而V则是与注意力权重用于当前位置的输出。

QK是专门用于求相似性的，如果只有一个向量比如k，而没有q，k 可以与其他位置向量的 k作相似性，但在自己单元却没有可以做相似性的地方，此时就无法进行各个相似性关系的比较从而得到关于全局位置的输出了，要解决该问题而再加一个向量q就可以实现了。

Self Attention 具体的流程图如下：

<img src="core algorithm.assets/image-20231001161954003.png" alt="image-20231001161954003" style="zoom: 50%;" />

我们先看图片所示$a^1$的例子，首先将$a^1$ 的 $q^1$ 与所有时间刻的 $k^1$ 进行inner product并除与 $\sqrt{d_k}$（ 其中$d_k$是Q和K的维度。）计算二者的相似性，得到 对应的相似性序列值 $a_{1,i}$  

> `为什么需要除与维度长度的根号呢？`
>
> 这是因为当数据维度值越大时，inner product的结果越大，通过将Q和K进行点积操作并除以$\sqrt(d_k)$来缩放注意力权重，这有助于减小梯度在计算注意力时的变化范围（维度越大值越大），使得训练更加稳定。

<img src="core algorithm.assets/image-20230922154628040.png" alt="image-20230922154628040" style="zoom:50%;" />

对相似性序列值 $a_{1,i}$  进行 Softmax 操作得到每个时刻的相似性权重 

<img src="core algorithm.assets/image-20230922155120587.png" alt="image-20230922155120587" style="zoom:50%;" />

而后通过对每个时间刻的相似性权重和Value向量点积累加，最终得到  $a^1$所对应的 $b^1$

<img src="core algorithm.assets/image-20230922160608376.png" alt="image-20230922160608376" style="zoom: 50%;" />

以此类推计算不同位置对应的$b_i$，输出 序列 $b$，我们可以得到矩阵相乘的式子，从而一次计算出所有时刻的输出，这样便实现了平行化计算（矩阵运算），计算过程如下图所示

<img src="core algorithm.assets/image-20230924110646423.png" alt="image-20230924110646423" style="zoom:50%;" />

这样我们就最终得到Self Attention 公式如下：
$$
Attention(Q, K, V) = softmax(\frac{QK^T }{ \sqrt{(d_k)}}) * V
$$
总的来说，Self Attention能够很好的解决前面的两个问题，但随之而来的也有两个问题，

1. 如何捕捉多种不同模式下的状态？

2. 在使用self-attention训练过程中，是通过各个位置的相似性权重加权求和得到对应序列，并没有显式的顺序信息（例如没有循环神经网络的迭代操作），我们保留输入序列的位置信息&顺序关系。

   这显然是不对的，我们需要对 Decoder 的输入进行一些处理，**即在训练中只注意当前训练时间刻前的历史数据**

下文所说的多头注意力机制和位置编码便是用于解决这两个问题

##### 多头注意力机制（Multi-head Self-attention)

为了实现捕捉多种不同模式下的状态，叠加self-attention便可以实现，每个self-attention模块被称为一个头（head）。通过并行计算多个头，模型可以学习不同粒度和关注不同方面的特征表示。

<img src="core algorithm.assets/image-20231001161940890.png" alt="image-20231001161940890" style="zoom:50%;" />

这里以两个头为例

**前面定义的一组 Q,K,V 可以让一个词 attend to 相关的词**，我们可以定义多组 Q,K,V，让它们分别关注不同的上下文。计算 Q,K,V 的过程还是一样，只不过线性变换的矩阵从一组 (WQ,WK,WV) 变成了多组 (W0Q,W0K,W0V) ，(W1Q,W1K,W1V)，… 如下图所示

<img src="core algorithm.assets/c7wEjO.png#shadow" alt="img" style="zoom: 33%;" />

对于输入矩阵 X，每一组 Q、K 和 V 都可以得到一个输出矩阵 Z。如下图所示

<img src="core algorithm.assets/c7weDe.png#shadow" alt="img" style="zoom: 50%;" />

最后加多个头的输出拼接再乘以一个**矩阵降维**得到同样对应维度的序列$b^i$。

<img src="core algorithm.assets/image-20230924113010453.png" alt="image-20230924113010453" style="zoom:50%;" />

参考总结：

- 目前的 Transformer 结构，encoder 端的 **head 是存在冗余的**，Multi-Head 其实不是必须的，或者丢掉一些 head 反而能提升性能
- 结构信息对文本建模很重要，无论是什么方法即使是 Transformer；
- 目前对 Self Attention 的改造还比较浅层，不够大刀阔斧。

#### 位置编码（Positional Encoding）

众所周知，文本是时序型数据，词与词之间的**顺序关系**往往影响整个句子的含义。举个栗子：

*计算机魔术师/是/一个/AI/盐就人员。一个/盐就人员/是/计算机/魔术师？计算机/魔术师/是/一个/盐就人员？？*

为了避免不必要的误会，所以我们在对文本数据（或者其他时序型数据）进行建模的时候需要保留输入序列的位置信息&顺序关系，此时一般需要引入位置编码。

`要建模文本中的顺序关系必须要用positional encoding吗？`

答案是No！

只有当我们使用**对位置不敏感(position-insensitive)的模型**对文本数据建模的时候，才需要额外使用positional encoding。

`那么什么是对位置敏感的模型？什么又是对位置不敏感的模型？`

如果模型的输出会随着输入文本数据顺序的变化而变化，那么这个模型就是关于位置敏感的，反之则是位置不敏感的。

用更清晰的数学语言来解释。设模型为函数y=f(x),其中输入为一个词序列x={ $ x_ {1} $ , $ x_ {2} $ , $ \cdots $ , $ x_ {n} $ }
输出结果为向量y。对x的任意置换x'={ $ x_ {k} $ , $ x_ {k_ {2}} $ , $ \cdots $ , $ x_ {kn} $ },都有f(x)=f(x) 则模型是关于位置不敏感的。

在我们常用的文本模型中，RNN和textCNN都是关于位置敏感的，使用它们对文本数据建模时，模型结构天然考虑了文本中词与词之间的顺序关系。而以attention为核心的transformer则是位置不敏感的，使用这一类位置不敏感的模型的时候需要额外加入positional encoding引入文本中词与词的顺序关系。

对于transformer模型的positional encoding有两种主流方式：

##### 绝对位置编码

现在普遍使用的一种方法**Learned Positional Embedding编码绝对位置**，相对简单也很容易理解。对不同的位置随机初始化一个不同的postion embedding，加到word embedding上输入模型，作为参数进行训练。

![图片](https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/640)

<img src="core algorithm.assets/image-20230924164520197.png" alt="image-20230924164520197" style="zoom:50%;" />

如图, 其中权重$W$ 可以分为 $W^I$ $W^P$ 对应序列 $x_i$ 和位置编码 $p^i$，矩阵计算后得到$a^i$ , $e^i$ 二者相加

##### 相对位置编码

使用绝对位置编码，不同位置对应的positional embedding固然不同，但是位置1和位置2的距离比位置3和位置10的距离更近，位置1和位置2与位置3和位置4都只相差1，这些关于位置的**相对含义**模型能够通过绝对位置编码get到吗？使用Learned Positional Embedding编码，位置之间没有约束关系，我们只能期待它隐式地学到，是否有更合理的方法能够显示的让模型理解位置的相对关系呢？

所以就有了另一种更直观地方法——相对位置编码。下面介绍两种编码相对位置的方法：Sinusoidal Position Encoding和Complex embedding。

###### **Sinusoidal Position Encoding**

理想情况下，信息编码（piece of information）的设计应该满足以下条件：

-  它应该为每个字（时间步）输出唯一的编码
-  不同长度的句子之间，任何两个字（时间步）之间的差值应该保持一致
-  我们的模型应该无需任何努力就能推广到更长的句子。它的值应该是有界的。
-  它必须是确定性的

使用正余弦函数表示绝对位置，通过两者乘积得到相对位置：

$$
PE_{(pos, 2i)} = \sin\left(\frac{{pos}}{{10000^{2i/d_{\text{model}}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{{pos}}{{10000^{2i+1/d_{\text{model}}}}}\right)
$$

其中，$pos$表示输入序列中的位置，$i$表示位置编码中的维度索引，$d_{\text{model}}$表示Transformer模型的隐藏单元大小。

这样设计的好处是**位置的psotional encoding可以被位置线性表示，反应其相对位置关系。**以下是原论文的引用：

>  We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed  offset  k, PEpos+k can  be represented as a linear function  of PEpos.

其中说到对于任何固定偏移 k，PEpos+k 可以表示为 PEpos 的线性函数。[这篇文章][https://kazemnejad.com/blog/transformer_architecture_positional_encoding/]就很好的讲解原理，对应添加的固定偏移offset可以通过PEpos 线性变换 dot product 一个矩阵M得到偏移后offset后的结果PEpos+k，这就包含了二者相对信息, 其中矩阵M如下：

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/image-20240219154010795.png" alt="image-20240219154010795" style="zoom:50%;" />

Sinusoidal Position Encoding虽然看起来很复杂，但是证明可以被线性表示，只需要用到高中的正弦余弦公式：****
$$
\sin (\alpha +\beta )=\sin \alpha \cdot \cos \beta +\cos \alpha \cdot \sin \beta \\ \cos (\alpha +\beta )=\cos \alpha \cdot \cos \beta -\sin \alpha \cdot \sin \beta
$$

对于位置pos+k的positional  encoding
$$
PE_{(pos+k,2i)}  =  \sin(w_{i}\cdot(pos+k)) =  \sin(w_{i}pos)\cos(w_{i}k)+\cos(w_{i}pos)\sin(w_{i}k) \\
 PE_{(pos+k,2i+1)}  =  \cos(w_{i}\cdot(pos+k)) =  \cos(w_{i}pos)\cos(w_{i}k)-\sin(w_{i}pos)\sin(w_{i}k)
$$
其中 $ w_ {i} $ = $\frac{1}{10000^{2i/d_{\text{model}}}}$

将公式稍作调整, 就有


对于位置的positional encoding



其中

将公式（5）（6）稍作调整，就有



注意啦，和相对距离是常数，所以有



其中为常数。

所以可以被线性表示。

计算和的内积，有



其中.

$PE_(pos+k)$和$PE_(pos)$的内积会随着相对位置的递增而减小，从而表征位置的相对距离。但是不难发现，由于距离的对称性，Sinusoidal Position Encoding虽然能够反映相对位置的距离关系，但是无法区分方向/(ㄒoㄒ)/~~

$$
PE_{pos+k}PE_{pos} = PE_{pos-k}PE_{pos}
$$


更加直观的对其可视化，可以看到图像关于对称，无法区分前后关系。

<img src="https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qEYdbiabicy0bnXo81GdEbQibKDII4jDFiauzFibQfZCMPo3FgALTtWe0ZIwwTUOyU16yxDtEhxib7413fg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

> 相比起拼接（concatenate ）位置编码 ，直接相加似乎看起来会被糅合在输入中似乎位置信息会被擦除，我们可以假设拼接位置编码，设为向量$p_i$ 代表其位置信息，
>
> Learned Positional Embedding ，，所以这里其实也是可以看作直接相加了位置编码（学习过的），根据研究表明这个W^P^ learn 有人做过了在convolution中`seq to seq`中类似的学习参数做法效果并不是很好，还有说其实会添加很多的不必要的参数学习等（issue地址：https://github.com/tensorflow/tensor2tensor/issues/1591，https://datascience.stackexchange.com/questions/55901/in-a-transformer-model-why-does-one-sum-positional-encoding-to-the-embedding-ra



这个公式中的分数部分将位置$pos$进行了缩放，并使用不同的频率分母（$10000^{2i/d_{\text{model}}}$）来控制不同维度的变化速度。这样，**不同位置和不同维度的位置编码会得到不同的数值，形成一个独特的向量表示**，

1. 周期性: 使用正弦和余弦函数能够使位置编码具有周期性。使得位置编码的值在每个维度上循环变化。这对于表示序列中的不同位置非常重要，因为不同位置之间可能存在重要的依赖关系。
2. 连续性: 正弦和余弦函数在输入空间中是连续的。这意味着相邻位置之间的位置编码也是连续的，有助于保持输入序列中的顺序信息的连贯性。
3. 维度关联: 位置编码中的维度与Transformer模型的隐藏单元大小相关联。这意味着不同维度的位置编码会以不同的频率变化，从而能够捕捉不同尺度的位置信息。较低维度的位置编码可以更好地表示较短距离的依赖关系，而较高维度的位置编码可以更好地表示较长距离的依赖关系。

<img src="file://D:/Study/Python全栈开发/人工智能/人工智能笔记本&案例/机器学习笔记本&案例/深度学习笔记本&案例/通用学习笔记/core algorithm.assets/gFfQoR.png%23shadow?lastModify=1708326365?lastModify=1708326365" alt="img" style="zoom: 33%;" />

正弦位置编码的另一个特点是论文假设模型可以**毫不费力地关注相对位置**。

参考文章：
如何优雅地编码文本中的位置信息？三种positioanl encoding方法简述： https://mp.weixin.qq.com/s/ENpXBYQ4hfdTLSXBIoF00Q

#### Padding Mask

Self Attention 所得到的序列输出 $b^i$ 看到了全部的输入序列，但是如果说不想考虑到全部的输入，比如时序预测中只能看到历史数据，只需要在计算未来时刻数据的相似性权重时，即给q , v 相似度输出默认给一个非常大的负数，在经过`softmax`就会变成0使得加权求和时避免加入未来时刻数据。

其中Transformers在output输入的多头注意力机制中可以看到添加了Masked（Padding Mask），

1.  传统 Seq2Seq 中 Decoder 使用的是 RNN 模型，因此在训练过程中输入 t 时刻的词，循环神经网络是时间驱动的，只有当 t 时刻运算结束了，才能看到 t+1 时刻的词。但在使用self-attention训练过程中，整个 ground truth都暴露在 Decoder 中，这显然是不对的，我们需要对 Decoder 的输入进行一些处理，**即在训练中只注意当前训练时间刻前的历史数据**
2.  句子长度不同中，根据最长句子补齐后，**对不等长的句子进行mask**。

> 为了屏蔽后续时间步，将Mask矩阵中对应位置的元素设置为一个很大的负数（例如-1e9），这样在经过Softmax函数后，注意力权重接近于零，相当于忽略了后续时间步的信息。

假设有一个时间序列数据，长度为10，你想要关注前6个时间步的数据，而忽略后面的时间步。

```
时间序列: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Mask矩阵:
[[0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9],
 [0, 0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9],
 [0, 0, 0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9],
 [0, 0, 0, 0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9],
 [0, 0, 0, 0, 0, -1e9, -1e9, -1e9, -1e9, -1e9],
 [0, 0, 0, 0, 0, 0, -1e9, -1e9, -1e9, -1e9]]
```

这样，在进行注意力计算时，将**输入序列与Mask矩阵相加**，之后再做 softmax，就能将 - inf 变为 0，得到的这个矩阵即为每个字之间的权重，从而达到忽略后续时间步的效果。

> 在不等长句子处理中则同理对无效句子进行masking操作即可



<img src="core algorithm.assets/Jy5rt0.png#shadow" alt="img" style="zoom:50%;" />



#### Transformer 输入输出细节

在机器翻译任务中，一个样本是由原始句子和翻译后的句子组成的。比如原始句子是： “我爱机器学习”，那么翻译后是 ’i love machine learning‘。 则该一个样本就是由“我爱机器学习”和 "i love machine learning" 组成。

这个样本的原始句子的单词长度是length=4,即‘我’ ‘爱’ ‘机器’ ‘学习’。经过embedding后每个词的embedding向量是512。那么“我爱机器学习”这个句子的embedding后的维度是[4，512 ] （若是批量输入，则embedding后的维度是[batch, 4, 512]）。

**padding**

因为每个样本的原始句子的长度是不一样的，此时padding操作登场了，假设样本中句子的最大长度是10，那么对于长度不足10的句子，需要补足到10个长度，shape就变为[10, 512], 补全的位置上的embedding数值自然就是0了

**Padding Mask**

在较短的序列后面填充0到长度为N进行补齐之后。对于那些补零的数据来说，我们的attention机制不应该把注意力放在这些位置上，此时添加Mask矩阵，使得这些位置的值加上一个非常大的负数(负无穷)，这样经过softmax后，这些位置的权重就会接近0。

> Transformer的padding mask实际上是一个张量，每个值都是一个Boolean，值为false的地方就是要进行处理的地方。

得到补全后的句子embedding向量后，直接输入encoder的话，那么是没有考虑到句子中的位置顺序关系的。此时需要再加一个位置向量，位置向量在模型训练中有特定的方式，可以表示每个词的位置或者不同词之间的距离；总之，核心思想是在attention计算时提供有效的距离信息。

关于positional embedding ，文章提出两种方法：

1.Learned Positional Embedding ，这个是绝对位置编码，即直接对不同的位置随机初始化一个postion embedding，这个postion embedding作为参数进行训练。

2.Sinusoidal Position Embedding ，相对位置编码，即三角函数编码。

下面详细讲下Sinusoidal Position Embedding 三角函数编码。

Positional Embedding和句子embedding是add操作，那么自然其shape是相同的也是[10, 512] 。

Sinusoidal Positional Embedding具体怎么得来呢，





1. 
2. 
1. 输入（Input）：
   - 源语言句子：将源语言句子进行编码，通常使用词嵌入（Word Embedding）来表示每个单词。例如，将英文句子"Hello, how are you?"转换为一系列词嵌入向量。
   - 位置编码（Positional Encoding）：为了捕捉单词在句子中的位置信息，Transformer模型引入位置编码，将位置信息与词嵌入向量相结合。
   - 输入嵌入（Input Embedding）：将词嵌入向量和位置编码向量相加，得到每个单词的最终输入表示。
2. 输出（Output）：
   - 目标语言句子：目标语言句子也会进行类似的处理，将目标语言句子进行编码和嵌入表示。
   - 解码器输入（Decoder Input）：解码器的输入是**目标语言句子的编码表示**，通常会在每个目标语言句子的开头添加一个特殊的起始标记（例如\<start>）来表示解码器的起始位置。
   - 解码器输出（Decoder Output）：解码器的输出是对目标语言句子的预测结果，通常是一个单词或一个单词的词嵌入向量。解码器会逐步生成目标语言句子，每一步生成一个单词，直到遇到特殊的结束标记（例如\<end>）或达到最大长度。

下面是一个机器翻译任务的例子：

源语言句子（英文）： "Hello, how are you?"
目标语言句子（法文）： "Bonjour, comment ça va ?"

在这个例子中，输入是源语言句子的编码表示，输出是目标语言句子的解码器输入和解码器输出。

输入（Input）：
- 源语言句子编码：[0.2, 0.3, -0.1, ..., 0.5] （词嵌入向量表示）
- 位置编码：[0.1, 0.2, -0.3, ..., 0.4]
- 输入嵌入：[0.3, 0.5, -0.4, ..., 0.9]

输出（Output）：
- 解码器输入：[\<start>, 0.7, 0.2, -0.8, ..., 0.6]
- 解码器输出：[0.1, 0.5, -0.6, ..., 0.2]

通过训练，Transformer模型会根据输入的源语言句子和目标语言句子进行参数优化，使得模型能够生成准确的目标语言翻译。

需要注意的是，具体任务中的输入和输出的表示方式可能会有所不同，这只是一个简单的机器翻译示例。不同任务和模型架构可能会有不同的输入和输出定义。

参考文章：
https://zhuanlan.zhihu.com/p/166608727

#### 一些值得思考的问题

##### 为什么说 Transformer 在 seq2seq 能够更优秀？

RNN等循环神经网络的问题在于**将 Encoder 端的所有信息压缩到一个固定长度的向量中**，并将其作为 Decoder 端首个隐藏状态的输入，来预测 Decoder 端第一个单词 (token) 的隐藏状态。在输入序列比较长的时候，这样做显然会损失 Encoder 端的很多信息，而且这样一股脑的把该固定向量送入 Decoder 端，**Decoder 端不能够关注到其想要关注的信息**。Transformer 通过使用Multi-self-attention 模块，让源序列和目标序列首先 “**自关联**” 起来，并实现全局观和并行能力，模型所能提取的信息和特征更加丰富，运算更加高效。  

并且从模型的复杂度分析来看如下：

![image-20231002105641906](core algorithm.assets/image-20231002105641906.png)

##### 关于代码

代码详解：http://nlp.seas.harvard.edu/2018/04/03/attention.html （Pytorch_实现，英文）
代码详解2：https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html（Pytorch实现，英文）
代码详解3：https://github.com/EvilPsyCHo/Attention-PyTorch（Pytorch实现，该仓库非常详细，非常建议学一学）
代码实现：https://github.com/foamliu/Self-Attention-Keras （Keras实现，结论：在这个经典的imdb数据集上的表现，只是中等。原因大致是该架构比较复杂，模型拟合需要更多的数据和轮次）
官方代码地址： https://github.com/tensorflow/tensor2tensor

如果有能力的话，大家可以尝试一下手撕代码哦。

参考文献:
https://wmathor.com/index.php/archives/1438/
https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=62
https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click&vd_source=2700e3c11aa1109621e9a88a968cd50c
https://wmathor.com/index.php/archives/1453/#comment-2101
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
http://jalammar.github.io/illustrated-transformer/
https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%887%EF%BC%89Mask%E6%9C%BA%E5%88%B6/#xlnet%E4%B8%AD%E7%9A%84mask
https://zhuanlan.zhihu.com/p/26753131

#### 扩展模型

下面是一些对Transformer模型进行改进和扩展的其他模型：

| 模型名称                                                     | 主要特点和解决的问题                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| BERT (Bidirectional Encoder Representations from Transformers) | BERT通过使用Transformer模型进行双向预训练来解决自然语言处理（NLP）任务。它利用了双向上下文信息，可以更好地处理语义理解和生成任务，如命名实体识别、情感分析等。 |
| GPT (Generative Pre-trained Transformer)                     | GPT是一个基于Transformer的语言生成模型。它通过在大规模文本语料上进行无监督预训练，并使用自回归方式生成文本序列。GPT能够生成连贯、富有语义的文本，适用于任务如文本生成、机器翻译等。 |
| XLNet                                                        | XLNet通过引入自回归和自编码的目标函数来扩展BERT。它解决了BERT在处理长文本时无法利用上下文信息的问题，通过重新排列输入序列的顺序来建模全局依赖关系。XLNet在各种NLP任务中取得了优异的性能。 |
| RoBERTa (Robustly Optimized BERT Approach)                   | RoBERTa是对BERT进行改进和优化的模型。它通过对BERT的预训练过程进行改进，如更大的批量大小、更长的训练时间等，进一步提升了模型的性能和鲁棒性。RoBERTa在多个NLP任务上取得了领先水平的结果。 |
| GPT-2                                                        | GPT-2是GPT的改进版本，具备更多的参数和更大的模型规模。它能够生成更长、更具连贯性的文本，但也面临着参数量大、计算资源消耗高的挑战。GPT-2在文本生成和语言理解任务上取得了令人瞩目的成果。 |
| T5 (Text-to-Text Transfer Transformer)                       | T5是一个通用的文本到文本转换模型，它可以在各种NLP任务中进行端到端训练。T5的训练目标是将不同的NLP任务转化为相同的文本生成问题，通过统一的框架来解决不同任务，使得模型更加灵活和易于使用。 |
| GPT-3                                                        | GPT-3是GPT的进一步扩展，具备巨大的模型规模和参数量。它是目前为止最大的语言模型之一，能够生成高质量的文本、执行算术计算、进行机器翻译等多种任务。GPT-3在广泛的NLP任务和领域中展示了强大的语言理解和生成能力。 |
| ransformer-XL                                                | Transformer-XL是对传统Transformer模型的改进，用于解决长文本建模中的上下文依赖问题。传统的Transformer模型在处理长文本时受到长度限制，无法有效利用上下文信息。Transformer-XL通过使用相对位置编码和循环机制，使得模型能够捕捉更长的上下文依赖关系。它通过存储先前的隐藏状态，使得模型可以在不同的片段之间重用先前的上下文信息，从而更好地处理长文本序列。Transformer-XL在语言建模和文本生成任务中取得了显著的改进。 |

这些模型都是基于Transformer架构进行改进和扩展的，通过增加参数、改变预训练目标、引入新的训练技巧等方式，进一步提升了模型在自然语言处理任务中的性能和表现。

###  Comformer

Conformer是Google在2020年提出的语音识别模型，基于Transformer改进而来，主要的改进点在于Transformer在提取长序列依赖的时候更有效，而卷积则擅长提取局部特征，因此将卷积应用于Transformer的Encoder层，同时提升模型在长期序列和局部特征上的效果，实际证明，该方法确实有效，在当时的LibriSpeech测试集上取得了最好的效果。

Wenet是出门问问语音团队联合西工大语音实验室开源的一款面向工业落地应用的语音识别工具包，该工具用一套简洁的方案提供了语音识别从训练到部署的一条龙服务，Wenet目前在github上获得了上千个stars，其易用性深受用户好评。

### Linformer：线性复杂性的自我关注

28000+ star

https://paperswithcode.com/paper/linformer-self-attention-with-linear

### [Vision Transformers](https://paperswithcode.com/paper/separable-self-attention-for-mobile-vision)

## 生成式模型

### 生成式对抗网络（Generator Adversarial Networks）

传统神经网络需要一个人类科学家精心设计的成本函数来指导学习，无监督学习为了解决这一问题，利用生成式对抗网络（Generator Adversarial Networks）对机器进行对抗训练（Adversarial Training）成了关键答案。

生成式对抗网络（Generator Adversarial Networks）是一种面向无监督学习的神经网络：它带有一个发生器（Generator），从随机输入中生成某类假数据（比如，假的图片）；假数据和来自世界的真数据会一起输入一个判别器（Discriminator）中，等待判别器裁决。

两者的关系就像一个骗子和一个测谎者。判别器这位测谎者要不断优化自己，尽可能地识别出发生器生成的“假图像”，防止自己被骗；而生成器这个骗子为了瞒过判别器，也必须不断进步。在这种动态的对抗中，生成器会得到训练，最后开始生成非常真实的图片，这意味着生成器掌握了图像的特征，自己掌握成本函数——某种意义上，这就是无监督学习。

### VAE

# 无监督学习

> 重要性

Facebook 首席人工智能科学家、纽约大学教授 Yann LeCun、卷积神经网络（CNN, Convolutional Neural Nets）的发明人之一曾这样说道：“如果人工智能是一块蛋糕，强化学习（Reinforcement Learning）就是蛋糕上的一粒樱桃，而监督学习（Supervised Learning）是蛋糕外的一层糖霜，但无监督学习/预测学习（Unsupervised/Predictive Learning）才是蛋糕本身。目前我们只知道如何制作糖霜和樱桃，却不知道如何做蛋糕” ，从这我们可以足以看到无监督学习的重要性。**无监督学习是实现稳健和通用表示学习的重要垫脚石。**

> 无监督学习困难

其中，无监督学习的一大困难之处在于：**对不确定性的预测**。比如当你将一支笔直立在桌面上时，松开手的那一刻，你并不知道这只笔将会朝哪个方向倒下。如果系统回答这只笔会倒下，却判断错误了倒下的方向，我们需要告诉系统，**虽然你不是完全正确，但你的本质上是正确的，我们不会惩罚你**。`此时需要引入曲面的成本函数，只要系统回答在曲面之下的数据，都是正确的答案。` 

此外，并不总是**清楚理想的表示是什么**，以及是否有可能在没有额外监督或专门化特定数据模态的情况下学习这种表示。无监督学习最常见的策略之一是**预测未来、缺失或上下文信息**。这种预测编码的想法是数据压缩信号处理中最古老的技术之一。在神经科学中，预测编码理论表明大脑可以**预测不同抽象级别的观察结果** （这和我们生活高度抽象事情是一样的，比如我们没有经历过这件事，但通过想象这件事发生的感受）。无监督学习工作已经成功地利用这些想法通过预测相邻单词来学习单词表示，等等。

参考文章：
https://en.jiemian.com/article/1198715.html、



### 自监督学习

自监督学习是一种特殊的无监督学习方法，它利用数据自身的结构或信息进行训练。在自监督学习中，**数据被人工设计成具有某种隐含标签或任务**，然后模型通过解决这个任务来进行训练。关键思想是让模型无需手动标签即可学习数据表示。如通过**对抗学习、聚类等方法**，提取出一些高效、鲁棒、通用的特征，然后通过少量有监督的数据微调模型，让有表征能力的“通用特征”进一步升级成具有区分性的“专属特征”，那么它就可以用较少量的标记数据用于下游任务，以达到与没有自监督学习的模型相似或更好的性能。

> 自监督训练的目的就是使用大量没有标签的数据和少量带标签的数据训练模型，以实现降低标注成本、降低迁移成本的目的。

自监督学习步骤

> 1. 基于对数据的理解，以编程方式从未标记的数据**生成输入数据和标签**，
>
> 2. 预训练：训练encoder，获得一些高效、鲁棒、通用的特征表示，他们不对应任何具体的下游任务，仅仅提取一些自认为可靠的特征出来。
>
> 3. Fine-tune：使用预训练的模型作为初始权重来训练感兴趣的任务，将前面得到的“乐高积木”进一步组合成为“树枝”、“耳朵”、“四肢”等有区分性的专属特征。实现语音识别、声音事件检测、说话人识别、语音唤醒等等下游任务。
>
>    如果我们在第二步中使用带有手动标签的数据而不是自动生成的标签，那么它将受到监督预训练，称为迁移学习的一个步骤。

自监督学习在文本、图像/视频、语音和图形等多个领域取得了成功。本质上，自我监督学习挖掘未标记的数据并提高性能。就像 Yann Lecun 的人工智能大蛋糕的比喻一样，这种自监督学习（蛋糕）每个样本可以**吃数百万次**，而监督学习（糖霜）**只能吃 10 到 10,000 口**。也就是说，自监督学习比监督学习能够从每个样本中获得更多有用的信息。

人类生成的标签通常关注数据的特定视图。例如，我们可以只用“马”一词来描述草地上的一匹马的图像进行图像识别，并提供像素坐标进行语义分割。然而，数据中有更多的信息，例如，马的头和尾巴位于身体的另一侧，或者马通常在草地上（而不是在下面）。这些模型可以直接从数据中学习更好、更复杂的表示，而不是手动标签。更不用说手动标签有时可能是错误的，这对模型是有害的。一项实验研究表明，清理 PASCAL 数据集可以将 MAP 提高 13。即使没有与最先进的技术进行比较，我们仍然可以看到错误标签可能会导致性能更差。

> 此外，在许多预训练（迁移学习）场景中，数据标记成本高昂、耗时且劳动密集型。此外，监督学习方法需要针对新数据/标签和新任务使用不同的标签。更重要的是，事实证明，对于基于图像的任务（即图像识别、对象检测、语义分割），自监督预训练甚至优于监督预训练。
>
> **换句话说，直接从数据中提取信息比手动标注更有帮助。**根据任务的不同，现在或不久的将来可能不需要许多昂贵的标签来进行更先进的自我监督学习。

通常，当自监督模型发布时，我们可以下载预训练的模型（有许多在线可提供下载的预训练模型，如hugging face)。可以对预训练的模型进行微调，并将微调后的模型用于特定的下游任务。例如，最著名的自监督学习示例可能是 BERT。BERT 以自我监督学习方式对 33 亿个单词进行了预训练。我们可以针对文本相关任务（例如句子分类）对 BERT 进行微调，比从头开始训练模型所需的工作量和数据要少得多。

#### 面对生成学习的自监督

**恢复原始信息**
a．非自回归：屏蔽标记/像素并**预测屏蔽**的标记/像素（例如，屏蔽语言建模（MLM））
b．自回归：预测**下一个**标记/像素;

<img src="https://img-blog.csdnimg.cn/d355c82413d745f2b04eb8d94c1f5b83.png" alt="在这里插入图片描述" style="zoom:50%;" />

通过周围数据预测屏蔽输入是最早的自监督方法类别。这个想法实际上可以追溯到这句话，“你应该通过它所陪伴的人来认识一个单词。”——约翰·鲁珀特·费斯（John Rupert Firth，1957），一位语言学家。这一系列算法始于 2013 年文本领域的 word2vec。word2vec 的连续词袋 (CBOW) 的概念是通过其邻居来预测中心词，这与 ELMo和BERT 的掩码语言建模 (MLM)不同。这些模型都被归类为非自回归生成方法。主要区别在于，后来的模型使用了更先进的结构，例如双向 LSTM（用于 ELMo）和 Transformer（用于 BERT）。

> 在语音领域，Mockingjay屏蔽了连续特征的所有维度，而 TERA 屏蔽了特征维度的特定子集。在图像领域，OpenAI 应用了 BERT 机制。在图领域，GPT-GNN 还屏蔽了属性和边。这些方法都屏蔽了部分输入数据并试图将它们预测回来。

另一种生成方法是预测下一个标记/像素/声学特征（自回归）。在文本领域，GPT系列型号是该类别的先驱。APC和 ImageGPT 分别在语音和图像领域应用了相同的思想。有趣的是，由于相邻的声学特征很容易预测，因此通常要求模型预测后面序列中的标记（至少 3 个标记之外）。

自监督学习（尤其是 BERT/GPT）的巨大成功促使研究人员将类似的生成方法应用于图像和语音等其他领域。然而，对于图像和语音数据，生成屏蔽输入更困难，因为选择有限数量的文本标记比选择无限数量的图像像素/声学特征更容易。性能改进不如文本字段。因此，研究人员在接下来的会议中还开发了许多其他非生成方法。

#### 面对预测学习的自监督

基于数据的理解、聚类或数据增强来设计标签
a：预测上下文（例如，预测图像块的相对位置，预测下一个片段是否是下一个句子）
b：预测聚类每个样本
c：预测图像旋转角度

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/image-20240106205108487.png" style="zoom:50%;" />

主要思想是设计更简化的目标以避免数据生成。最关键和最具挑战性的一点是任务需要处于适当的难度级别才能让模型学习。

例如，在预测文本字段中的上下文时，BERT 和 ALBERT 都预测下一个片段是否是下一个句子。BERT 通过随机交换下一个片段与另一个片段（**下一句预测；NSP**）来提供负训练样本，而 ALBERT 通过交换上一个和下一个片段（**句子顺序预测；SOP**）来提供负训练样本。SOP 已被证明优于 NSP。

>  一种解释是，通过主题预测很容易区分随机句子对，以至于模型没有从 NSP 任务中学到太多东西；而SOP允许模型学习连贯关系。因此，需要领域知识来设计好的任务和实验来验证任务效率。

像 SOP 一样预测上下文的想法也应用于图像领域（预测图像块的相对位置）和语音领域（预测两个声学特征组之间的时间间隔）。

另一种方法是通过聚类生成标签。在图像领域，DeepCluster应用了k-means聚类。在语音领域，HuBERT 应用了 k 均值聚类，而 BEST-RQ 采用了随机投影量化器。

图像领域的其他任务有：通过图像的颜色通道预测灰度通道（反之亦然；），重建图像的随机裁剪块（即修复；），重建原始分辨率的图像 ，预测图像的旋转角度 ，预测图像的颜色并解决拼图游戏。

#### 面对对比学习的自监督

对比学习是为了在不关注样本全部细节的情况下，训练一个Encoder将样本转化为表征（representation，比如用一个编码器将数据编码成高维向量，就可以将得到的向量称为是数据的representation），使得representation包含了更显著的、重要的、有区分度的特征，学到这样的表示之后，用来帮助提升下游任务的性能。

> 对比学习的关键方法是根据对数据的理解生成正负训练样本对。模型需要学习一个函数，**使得两个正样本具有较高的相似度分数，两个负样本具有较低的相似度分数**。因此，适当的样本生成对于确保**模型学习数据的底层特征/结构**至关重要。

Encoder可以将高维数据**压缩到更紧凑**的潜在嵌入空间中进行编码的表示以学习对（高维）信号不同部分之间的底层共享信息，其丢弃了更局部的低级信息和噪声**更容易建模**。但是预测高维数据的挑战之一是均方误差和交叉熵等单峰损失不是很有用，如果直接建模源数据和特征的复杂关系导致除了模型计算量很大之外，提取 x 和 c 之间的共享信息可能不是最佳的。（例如，图像可能包含数千位信息，而类标签等高级潜在变量包含的信息要少得多（1024 个类别为 10 位）），我们就需要一个强大有效而精炼的Encoder编码器。

`但是对于一个encoder来说，怎么样的encoder是好的encoder呢？`

autoencoder的训练过程中要求encoder作用过后得到的latent vector被decoder解码后得到的新的matrix和raw matrix的reconstruction error尽可能小，更好的可解释性，更强的代表性和区分度。

> 所谓区分度，举个例子说明：有三个样本组成的集合{x，x+，x-}， x+ 表示和 x 相似的样本， x- 表示和 x 不相似的样本，“区分度”意味着，x 的representation和 x+ 的representation要较为相似，而 x 的representation和 x- 的representation要较不相似，那样的representation就是有区分度的。

`那么如何去判断encoder编码器足够好呢？`

这应该就是Contrastive对比学习的思路，我们通过引入一个**Discriminator**（判别器）来判断编码好坏，这里我们以Discriminator（判别器）为二分类判别器：

> 如下图，首先我们把原来的image数据和编码后的latent vector构建数据集，例如用(Xi, yi)，其中yi是Xi经过encoder后编码的latent vector。我们训练Discriminator，在面对(Xi,yi)告诉discriminator这是positive sample，你该给它高分（1分），然后我们把所有样本（常常是一个batch里的）和与之对应编码器编码的latent vector打乱，比如我们把(Xi,yj)配对放进discriminator，告诉discriminator，这是negetive sample， 应该给它们0分。
>
> <img src="https://img-blog.csdnimg.cn/c53036ae7f074cc79c6e3d627208f065.png" alt="在这里插入图片描述" style="zoom:50%;" />
>
> 训练好一部分数据后，我们拿一些discriminator没见过的组给它，让discriminator去做二分类，判断输入的(X,y)是不是对应的raw data和latent vector。如果discriminator的识别准确率很高，说明编码所编码的特征是具有分辨度的，否则说明我们很难将encoder学到的latent vectors分开，也就是说这些latent vector是**不够具有代表性**的。

这里讲这个例子是为了提到这种“把encoder好不好的问题转换为encoding出来的vector和原数据配对后 真假样本能不能被一个分类器分开”的思路，**即通过模型来验证**，我们认为好的latent vector能让classifier比较容易地**把正确的latent vector和随机的latent vector 分开。**

按照上面的思路，我们来理解一下对比学习的目标。

用 s(a, b) 表示计算 a 和 b 的相似度，f(·) 表示能将 x 转化为representation的映射函数，x+ 是相似样本，x- 是非相似样本，则对比学习的目标就是学习这个映射函数 f(·)，使得 f(·) 满足下面的式子：

$$
s(f(x),f(  x^ {+}  ))>>s(f(x),f(  x^ {-}  ))
$$


比较朴素的向量相似度的计算方式就是向量内积。我们假设s(a,b)表示a和b的内积，那么希望给定一个样本x，要使得x的representation和所有x+的representation的内积尽可能大，而x和所有x-的representation的内积尽可能小，那么就是下面公式的期望尽可能的大：

$$
\frac{e^{f(x)^{T}f(x^+)}}{e^{f(x)^{T}f(x^+)} +e^ {f(x)^ {T}f(x^ {-})} }
$$
根据最大化期望的目标，可以得到损失函数的形式如公式所示：
$$
E[-\log(\frac{e^{f(x)^{T}f(x^+)}}{e^{f(x)^{T}f(x^+)}+e^{f(x)^{T}f(x^-)}})]
$$
通常在实践中会设定每一轮优化时，采样N个样本，用1个相似本和N-1个不相似样本（可以是随机，根据分布）来计算损失，那么这个Loss就可以看做是一个 N 分类的交叉熵Loss，所以对比学习的损失函数又被表示成下面的公式(该损失函数在对比学习的文章中被称为 InfoNCE  Loss)：

$$
E[-\log(\frac{e^{f(x)^{T}f(x^+)}}{e^{f(x)^{T}f(x^+)}+\sum_{j=1}^{N-1}e^{f(x)^{T}f(x^-)}})] 
$$
该公式便是噪声对比估计（NCE）的损失函数，那么对上面的这个损失函数进行优化，就可以完成我们的最初的目标，也就是让x经过 f(·) 映射函数之后，得到的编码和 x+ 的编码相似度尽可能高，和 x- 的比编码相似度尽可能低。这是对比学习的一个通用的目标。其实这个优化过程和负采样的思路是相通的，做NLP的朋友应该熟悉word2vec词向量，word2vec有两个加速训练的方法，其中一个就是负采样，负采样可以i避免在整个词典上进行softmax时候计算量巨大的问题，而对比学习也是为了不对全局的特征进行建模，只关注重要的特征。

参考：
https://blog.csdn.net/sinat_34604992/article/details/108385380

##### CPC 奠基之作

CPC是谷歌 DeepMind团队2019年发布的一篇论文《**Representation Learning with Contrastive Predictive Coding**》。改论文提出了一种**通用**的无监督学习方法 —— 一个用于提取紧凑潜在表示以编码对未来观察的预测的框架，从高维数据中提取有用的表示，称之为对比预测编码（Contrastive Predictive Coding）。所得模型——对比预测编码（CPC）——应用于广泛不同的数据模态、图像、语音、自然语言和强化学习，并表明**相同的机制**可以学习这些领域中**每个领域的有趣的高级信息**，其**性能优于其他方法**。

###### **核心思想**

模型思想重点在于representation learning（找到raw data的好的representation vector（也叫hidden vector/ latent vector)，并希望这个representation有很好的predictive的能力。）

我们根据Contrastive的对比学习思想，原文的Discriminator判别器的作用是用于使得Encode编码的latent vector满足时序特征，并尽可能保留原始信息更多信息，具体来说**原文Discriminator判别器采用的模型是预测模型（自回归）来验证特征**。通过输入过去时刻z(t-1)、z(t-2)、z(t-3)、z(t-4) 的编码特征输入到自回归模型后得到的预测值c(t)经过变换，可以与未来时刻z(t+1)、z(t+2)、z(t+3)、z(t+4)的编码特征尽量的接近。**即，c(t)通过一些变换后，可以很好的用来重构未来的特征z(t+k)。**

> 在时间序列和高维建模中，自回归模型使用下一步预测的方法利用了信号的**局部平滑度**。当进一步预测未来时，共享信息量会变得更低，模型需要推断出**更多的全局结构**。这些跨越许多时间步骤的“**慢特征**”(通用特征）通常更有趣（例如，语音中的音素和语调、图像中的对象或书籍中的故事情节）。

我们首先看预测编码流程架构如下图（音频为例输入，原文对图像、文本和强化学习使用相同的设置）。首先Encoder将输入的观测序列 xt 映射到潜在表示序列 zt = Encoder(xt)，并将 论文里是一般性地用 gar 来表示这个可以做预测的有回归性质的model，通常大家会用非线性模型或者循环神经网络。<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/image-20240131205036998.png" alt="image-20240131205036998" style="zoom: 50%;" />

 我们用t时刻及其之前的若干时刻输入这个回归模型，自回归模型将总结潜在空间中的所有 z≤t时刻的编码特征并生成上下文潜在表示 ct = gar(z≤t)，即涵盖了对过去信息的memory的输出ct。

`此时我们希望ct是足够好的且具有预测性质的，那么如何评价它好不好呢？`

我们可以通过用ct去预测之后k个时刻的latent vector（k是我们感兴趣的预测的步长）这里记作$z^t+1,z^t+2,...,z^t+k$, 然后我们希望 z^t+i （预测的latent vector）和 zt+i (真实的 xt+i 的latent vector）尽可能相似。论文采用直接用使用简单的对数双线性模型对ct进行变换，通过 W1,W2,...,Wk 乘以 ct 做的预测，

fk(x)是z(t+k)和c(t)的相似性度量函数，可以是函数形式、可以是内积、也可以是余弦距离。z(t+k)是t时刻起，未来第k帧的潜在特征,每一个k都对应了一个fk(x)。xj∈Xn（n=1,2,3...N）参与loss计算的这个N个样本中，有1个正样本z(t+k)，和N-1个负样本，其中负样本是随机从其他时刻采样的值。整个损失函数的目的是使z(t+k)跟[W(k)c(t)]的相似度尽量的高，跟其他负样本的相似度尽量的低，这样loss才能尽可能的小。

然后用向量内积来衡量相似度。（也可以用非线性网络或循环神经网络）

这样就得到了论文所提出了的相似度函数

 $ f_ {k} $ ( $ x_ {t+k} $ , $ c_ {t} $ )=exp( $ z_ {t+k}^ {T} $ $ W_ {k} $ $ c_ {t} $ )

其中 fk()表示计算 ct 的预测和 xt+k （真实的未来值）符不符合。

`那么得到了相似度函数，我们要如何评价总体的特征编码效果好不好以设置损失函数呢？`

原论文提出了用了正样本分数分布与负样本分数比值来进行评价，公式： $ \frac {p(x_ {t+k|ct})}{p(x_ {t+k})} $  

> 假设给定了选取X={x1,x2,...,xN}共N个随机样本，包含了一个是来自于${p(x_ {t+k|ct})}$ 的正样本 ,和其他来自*proposal* distribution（提议分布）${p(x_ {t+k})}$的N-1个负样本negative sample（noise sample）

这个分布表示了模型编码的分数好坏，模型编码效果越好分数比值的值越大（越靠近1），而相似度函数中则是计算分数的，那么有公式 $ f_ {k} $ ( $ x_ {t+k} $ , $ c_ {t} $ ) $ \infty $ $ \frac {p(x_ {t+k|ct})}{p(x_ {t+k})} $ ，即**正样本的相似度函数**是和$ \frac {p(x_ {t+k|ct})}{p(x_ {t+k})} $ 是正相关的，也就是说fk()值越大相似度越大，预测效果越好，那么模型特征表征也就越好，那么设计损失函数便是将 $ \frac {p(x_ {t+k|ct})}{p(x_ {t+k})} $ 最大化，这样我们就得到了最终的损失函数表达如下（Noise-Contrastive Estimation(NCE) Loss, 在这篇文献里定义为InfoNCE）

$$
L_{N}=−E[log\frac{f_{k}(x_{t+k},c_{t})}{Σ_{x_{j}∈X}f_{k}(x_{j},c_{t})}]
$$
其中分母是负样本的分数之和（从分布中选取全部的负样本），最终的对数分布比期望越小时，$ \frac {p(x_ {t+k|ct})}{p(x_ {t+k})} $  越大。

> 看到一个很好的思路，这里引用以下	：
>
> `为什么负采样是在整段序列上进行采样，那样不是会采样到窗口内的单词吗？`
>
> 我们知道，正样本来源于 t 时刻的一定窗口内的单词，按照正常思路，负样本应该来源于窗口以外的单词，这里有一个问题，假如一段长的序列，窗口内的单词在窗口外也出现了（比如“你，我”等常见词），这仍然不能避免负采样取到窗口内单词。所以作者直接在整段序列上进行负采样，负样本来源于整段序列的分布，正样本来源于窗口内单词的分布，这样做是为了让模型在给定一个context情况下判断某个样本来源于窗口内分布还是整段序列的噪声分布，也就是只需要模型可以**区分窗口内分布和整段序列的噪声分布**，这其实是一种退而求其次的方法，因为负采样本身就是为了避免在整个词典上进行softmax的开销过大问题，假如纠结负采样会采样到真实样本，那么干脆直接不要负采样，就在整个词典上进行 正样本与其他单词的区分就好了（这样做显然是没必要的）。所以，CPC论文的负采样就直接在整段序列上进行采样，当序列长度足够长，且负采样的次数足够多时，这么做是能够很好的模拟真实噪音分布的，而CPC的论文实验部分也证明了这一点。

最终编码器和自回归模型都将经过训练，将共同优化基于 NCE 的损失。其中将通过自回归模型的训练来反向使优化编码器编码。

我们再来总结一下，为了训练得到有效的Encoder编码器，论文引入了一个Decriminator 自回归模型来验证编码好坏，其中使用了相似度函数和正负样本分数比来作为评价指标，通过同时训练自回归模型和编码器最终得到良好的Representation learning 表示学习。

###### **算法实践**

我们在训练好模型之后，可以使用的训练好encoder部分或者encoder+自回归模型提取数据的特征用于不同的下游任务，或者在模型后面串接一个小型的微调网络，然后通过cross entropy或者CTC等任务的损失函数即可实现下游任务的训练。

算法代码实现库：
算法的 Keras 实现：https://github.com/davidtellez/contrastive-predictive-coding
Github复现主题系列：https://github.com/topics/contrastive-predictive-coding

参考：CPC的效果并不如监督训练的效果好，但它引入了一个很好的自监督训练思想，才有了后面的里程碑似的成果——wav2vec2.0。

参考文章：
非常感谢这篇文章，给我了很大的思路和帮助：https://zhuanlan.zhihu.com/p/129076690
https://blog.csdn.net/sinat_34604992/article/details/108385380

##### 图像领域对比学习

图像领域的对比学习应用来自同一原始图像的**两种不同的数据增强来生成正样本对**，并使用两个**不同的图像作为负样本对**。

最关键和最具挑战性的两个部分是**增强的强度和负样本对的选择**。如果增强太强以至于同一样本的两个增强样本之间没有关系，则模型无法学习。同样，如果增强量太小以至于模型可以轻松解决问题，那么模型也无法为下游任务学习有用的信息。至于选择负样本对，如果我们随机分配两个图像作为负样本对，它们可能是同一类（例如，猫的两个图像），这会给模型带来冲突的噪声。如果负对很容易区分，那么模型就无法学习数据的底层特征/结构。对比学习最著名的例子是 SimCLR (v1，v2）和MoCo（v1，v2），总的来说就是要让任务足够困难让模型能够很好的学习到数据的底层特征/结构.

##### 音频领域对比学习

对于语音领域，一种方法是应用像 SimCLR ( Speech SimCLR ) 这样的增强。另一种方法是**使用相邻特征作为正对，使用不同样本的特征作为负对**（例如，CPC、 Wav2vec ( v1 , v2.0 ) 、VQ-wav2vec和Discret BERT）。

##### 文本领域对比学习

相对来说NLP领域的对比学习受到了CV领域的启示，整体发展比较零散，脉络感没有那么强，此处，我们主要选择了无监督领域比较有代表性的ConSERT[16]以及SimCSE[17]来进行介绍，对于有监督领域则主要介绍R-Drop[18]。

参考文章：

https://blog.csdn.net/chumingqian/article/details/131351085
https://zhuanlan.zhihu.com/p/39928037
https://zhuanlan.zhihu.com/p/528648578

# 多模态学习

人类使用五种感官来体验和解释周围的世界。我们的五种感官从五种不同的来源和五种不同的方式捕获信息。模态是指某事发生、经历或捕捉的方式。人工智能正在寻求模仿人类大脑。

多模态学习在这里不同与有无监督的划分了, 多模式深度学习是一个机器学习子领域，旨在训练人工智能模型来处理和发现不同类型数据（模式）之间的关系——通常是图像、视频、音频和文本。通过结合不同的模态，深度学习模型可以更普遍地理解其环境，因为某些线索仅存在于某些模态中。想象一下情绪识别的任务。它不仅仅是看一张人脸（视觉模态）。一个人的声音（音频模态）的音调和音高编码了大量关于他们情绪状态的信息，这些信息可能无法通过他们的面部表情看到，即使他们经常是同步的。

**模态表示**
多模态表示分为两类。

1. 联合表示: 每个**单独的模态**被编码，然后**放入一个相互**的高维空间。这是最直接的方式，当模态具有相似的性质时可能会很有效。
2. Coordinated representation:  每个单独的模态都被编码而**不考虑彼此**，但是它们的表示然后**通过施加限制来协调**。例如，它们的线性投影应该最大程度地相关

**模态融合**
融合是将来自两种或多种模式的信息结合起来以执行预测任务的任务。

- 由于多模态数据的异构性，多模态（如视频、语音和文本）的有效融合具有挑战性。

**模态对齐**
对齐是指识别不同模态之间的**直接关系**的任务。

**模态转换**
模态转换是将一种模态映射到另一种模态的行为。主要思想是如何将一种模态（例如，文本模态）转换成另一种模态（例如，视觉模态），同时保留语义。

 **模态共同学习**
多模式共同学习旨在将通过一种或多种模式学习的信息转移到涉及另一种模式的任务。在低资源目标任务、完全/部分缺失或嘈杂模式的情况下，共同学习尤为重要。

> **多模态应用场景**

**图像领域**

图像字幕是为给定图像生成简短文本描述的任务。这是一项多模态任务，涉及由图像和短文本描述组成的多模态数据集。它通过将视觉表示转换为文本表示来解决前面描述的翻译挑战。该任务还可以扩展到视频字幕，其中文本连贯地描述短视频。

对于将视觉模式转换为文本的模型，它必须捕获图片的语义。它需要检测关键对象、关键动作和对象的关键特征。参考图例。“一匹马（关键对象）背着（关键动作）干草（关键对象）的大负载（关键特征）和两个人（关键对象）坐在上面。” 此外，它需要推断图像中对象之间的关系，例如，“双层床下面有一个狭窄的架子（空间关系）。”

然而，正如已经提到的，多模态翻译的任务是开放式的和主观的。因此，标题“两个人骑着装满干草的马车”和“两个人用马车运送干草”也是有效的标题。

图像字幕模型可用于提供图像的文本替代方案，从而帮助盲人和视障用户, 或是将图像的状态用文本快速描述, 如后厨行为识别,

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/6f7dfdd2fe6c48ca8875bce05c4afcf6.png" alt="在这里插入图片描述" style="zoom: 33%;" />



多模态数据集
没有数据，就没有学习。

多模态机器学习也不例外。为了推进该领域，研究人员和组织已经创建并分发了多个多模式数据集。以下是最受欢迎的数据集的完整列表：

医疗数据链接

5.1 COCO-Captions 数据集：
一个多模式数据集，包含 330K 图像并附有简短的文本描述。该数据集由微软发布，旨在推进图像字幕研究。

5.2 VQA
：一个视觉问答多模态数据集，包含 265K 个图像（视觉），每个图像至少有三个问题（文本）。这些问题需要理解视觉、语言和常识知识才能回答。适用于视觉问答和图像字幕。

5.3CMU-MOSEI
：Multimodal Opinion Sentiment and Emotion Intensity (MOSEI) 是一个用于人类情绪识别和情绪分析的多模态数据集。它包含 23,500 个句子，由 1,000 名 YouTube 演讲者发音。该数据集将视频、音频和文本模式合二为一。用于在三种最流行的数据模式上训练模型的完美数据集。

5.4.Social-IQ
：一个完美的多模态数据集，用于训练视觉推理、多模态问答和社交互动理解方面的深度学习模型。包含 1250 个音频视频（在动作级别），并带有与每个场景中发生的动作相关的问题和答案（文本）。

5.5 Kinetics 400/600/700
：此视听数据集是用于人类动作识别的 Youtube 视频集合。它包含人们执行各种动作的视频（视觉模态）和声音（音频模态），例如播放音乐、拥抱、进行运动等。该数据集适用于动作识别、人体姿势估计或场景理解。

5.6.RGB-D 对象数据集
：结合了视觉和传感器模态的多模态数据集。一个传感器是 RGB，对图片中的颜色进行编码，而另一个是深度传感器，对物体与相机的距离进行编码。该数据集包含 300 个家庭物品和 22 个场景的视频，相当于 250K 张图像。它已被用于 3D对象检测或深度估计任务。
其他多模式数据集包括IEMOCAP、CMU-MOSI、MPI-SINTEL、SCENE-FLOW、HOW2、COIN和MOUD

#### **多模态架构**

多模态神经网络通常是多个单模态神经网络的组合。例如，视听模型可能由两个单峰网络组成，一个用于视觉数据，一个用于音频数据。这些单峰神经网络通常分别处理它们的输入。这个过程称为**编码**。在进行单峰编码之后，必须将从每个模型中提取的信息**融合**在一起。已经提出了多种融合技术，范围**从简单的连接到注意机制**。多模态数据融合过程是最重要的成功因素之一。融合发生后，最终的“决策”网络接受融合后的编码信息，并接受最终任务的训练。

简单来说，多模态架构通常由三部分组成：

1. 编码单个模态的单峰编码器。通常，每个输入模式一个。
2. 一种融合网络，在编码阶段结合从每个输入模态中提取的特征。
3. 接受融合数据并进行预测的分类器。

我们将以上称为编码模块（下图中的 DL 模块）、融合模块和分类模块。

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/5d3971c210d04714930465b83719c842.png" alt="在这里插入图片描述" style="zoom:67%;" />

##### 模态编码

在编码过程中，我们寻求创建有意义的表示。通常，每个单独的模态由不同的单模态编码器处理。但是，通常情况下输入是嵌入形式而不是原始形式。

例如，word2vec 嵌入可用于文本;
而 COVAREP(一个基于matlab开发的语音库) 嵌入可用于音频。
多模态嵌入，如data2veq，将视频、文本和音频数据转换为高维空间中的嵌入，是最新的实践之一，并且在许多任务中优于其他嵌入，实现了 SOTA 性能。

> 通常当模态**在本质上是相似**时，使用**联合表示的方法效果很好**，并且是最常用的方法。

在设计多模态网络的实践中，编码器的选择是**基于在每个领域表现良好的编码器**，
因为**更加强调融合方法的设计**。

- 许多研究论文使用经典的 ResNets 作为视觉模式，
- 使用RoBERTA作为文本。

##### 模态融合

融合模块负责在特征提取完成后组合每个单独的模态。用于融合的方法/架构可能是成功的最重要因素。

最简单的方法是使用简单的操作，例如连接或求和不同的单峰表示。

> 然而，更具有经验和成功的方法，已经被研究出来了。例如，**交叉注意力层机制**是最近成功的融合方法之一。它已被用于以更有意义的方式捕获跨模态交互和融合模态。

下面的等式描述了交叉注意力机制，并假设您基本熟悉自注意力。

 $ \alpha_{kl}  = s(  \frac {K_ {1}Q_ {k}}{\sqrt {d}}  )  V_ {l} $ 

 表示注意力得分向量，s(.)表示softmax函数，K、Q和V分别是注意力机制的Key、Query和Value矩阵。
为了对称, $ \alpha_{lk} $ 也被计算出来，并且可以将**两者相加以创建一个注意力向量**，该向量映射所涉及的**两种模 态（k，l）**之间的协同作用。

本质上, $ \alpha _ {kl} $ , $ \alpha _ {lk} $ ,两者之间的区别
 $ \alpha _ {kl} $ 是模态k被用作query, 并且模态l扮演键和值的角色。
 $ \alpha_{lk} $ 是模态l被用作query, 并且模态k扮演键和值的角色。

在三个或更多模态的情况下，可以使用多个交叉注意机制，以便计算每个不同的组合。
例如，如果我们有视觉 (V)、文本 (T) 和音频 (A) 模态，那么我们创建组合 VT、VA、TA 和 AVT 以捕获所有可能的跨模态交互。

即使在使用注意力机制之后，也经常执行上述跨模态向量的串联以产生融合向量F。Sum(.)、max(.) 甚至池化操作也可以代替使用。



参考文章:

https://www.v7labs.com/blog/multimodal-deep-learning-guide

https://blog.csdn.net/chumingqian/article/details/131293445

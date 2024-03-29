## 过拟合问题解决方案

正则化的目的就是为了解决 **过拟合**的问题，就是说网络没有很好的学习到数据的表达模式，只学习到仅和训练数据有关的模式中错误的或者无关紧要的模式（过拟合） ，以下都是关于该办法的一些解决方案

1.  叫小网络大小
2.  添加权重正则化
3.  添加更多的训练数据
4.  添加Dropout

### 减小网络大小

在深度学习中，模型可以学习到的参数叫做模型的容量，如果模型参数越多，将拥有更多的记忆容量，这将使得模型非常轻松对数据进行输入与结果的映射，而没有学习到数据相关的模式，泛化能力差，这就是一个过拟合的表现。比如说在一个简单的任务里，我们使用很大容量的模型就会造成强映射，而不能再出现新数据的时候进行泛化，比如NMIST数据集，每5000个参数对应一个数字，50000个参数十个数字不重复，虽然精度和loss很好，但是没有很好的泛化能力，这个时候我们应该使用较小的参数和层数进行训练，所以设定模型一般都是由少到多。

### 添加权重正则化

奥卡姆剃刀原理：

>  奥卡姆剃刀原理（Occam's Razor(剃刀)）是一种基于简洁性和经济性的哲学原则，常用于科学推理和问题求解中。该原理最早由威廉·奥卡姆（William of Ockham）提出，他是一位14世纪的英国修道士和哲学家。
>
>  奥卡姆剃刀原理的核心思想是：**在面对多个解释或假设时，应该优先选择最简单、最经济、最少假设的解释或假设。**简单来说，它主张在没有足够证据支持的情况下，应该**倾向于选择更简单的解释或理论**。
>
>  原理的名称来源于一个寓言式的比喻：如果你看到一只马在草地上奔跑，不需要引入额外的假设来解释它的行为。例如，你不需要假设有一个无形的独角兽或一群外星人在追逐马。最简单的解释是，马自己在奔跑。
>
>  在科学推理中，奥卡姆剃刀原理被广泛应用。当面临多种理论或解释时，科学家通常倾向于选择那些假设少、简洁性高，并能够解释已知观测结果的理论。这是因为**简单的理论更容易验证、推断和应用**，而且不容易引入**不必要**的复杂性。
>
>  奥卡姆剃刀原理并不意味着简单就一定正确，或者复杂就一定错误。它只是一种启发式原则，用于帮助科学家在多个理论或解释之间进行选择时，倾向于选择最简单和经济的解释。然而，当有足够的证据表明较复杂的解释更准确时，奥卡姆剃刀原理并不排除选择复杂的解释。

如果说权重的大小范围很大，那就会容易出现复杂的过拟合现象，可以想象 0 ~ 520 与 0 ~ 1314 哪个组合更多，而且权重很大还会让模型只注重学习某一些特征或样本的，导致泛化效果差。

 **L1正则化或L2正则化**

L1正则化又称为lasso回归(Lasso Regression)，L2正则化又称为岭回归(Ridge Regression)，也成为**权重衰减**。

正则化的alpha值一般需要通过实验调参来确定，通常使用交叉验证等技术来选择最优的超参数。不同的问题可能需要不同的alpha值。

L1正则化的公式为：$\lambda \sum_{i=1}^{n}|w_i|$

L2正则化的公式为：$\lambda \sum_{i=1}^{n}w_i^2$

其中，$w_i$表示模型参数，$\lambda$表示正则化参数。L1正则化会使模型的参数更加稀疏，适合于特征选择；L2正则化则可以有效防止过拟合，适合于参数的约束。

选择使用哪种正则化（L1正则化或L2正则化）通常取决于具体情况。一般来说，L2正则化常常能够更好地控制权重的大小，从而防止过拟合。而L1正则化则更倾向于将一些特征的权重归零（根据导数来看，权重在正则化函数会减1），这有助于特征选择和模型的解释性。 我们还可以同时做L1和L2正则化。

当使用`Keras`库进行L1正则化（Lasso）和L2正则化（Ridge）的模型训练时，你可以通过在层的参数中指定相应的正则化项来实现。以下是使用`Keras`进行L1正则化、L2正则化以及同时使用L1和L2正则化的模板代码示例：

`keras `模板代码：

```python
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=100, activation='relu',
                kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(64, activation='relu',
                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1, activation='sigmoid'))

# 模型的编译和训练
# ...
```

### Dropout

《 Dropout: A Simple Way to Prevent Neural Networks from Overfitting 》 

Hinton 发明该方法灵感之一是银行的防欺诈机制、集成学习和无性有性繁殖 ，建议可以去看原论文

>  以集成学习思想为例：
>
>  在每次训练迭代中，随机地将一部分神经元的输出置为零（即丢弃），并将剩下的神经元的输出**按比例缩放**。这样做的效果相当于训练了多个不同的模型，每个模型只使用了部分神经元，从而实现了类似于集成学习的效果。
>
>  通过Dropout，网络中的每个神经元都有可能在训练过程中被丢弃，因此**不同的神经元组合可以得到不同的子模型**。最终的预测结果是这些子模型的预测结果的平均值或投票结果。，增加了模型的多样性。这样一来，网络**不会过度依赖于任何一个神经元**，提高了模型的鲁棒性和泛化能力。

为了防止过拟合，除了可以使用正则化技术或者在模型中**加入一些随机性**。其中一种常见的**随机性技术**就是 Dropout，它在训练过程中**随机地将一部分神经元输出置为0**，这样可以减少神经元之间的依赖关系，降低模型的复杂度，提高模型的泛化能力。

在前向传播过程中，对于**每个神经元的输出**，以概率 p 将其置为0，以概率 1-p 将其保留。在反向传播时，被置为0的神经元对损失函数的贡献为0，**因此不会对参数更新产生影响**。**在模型预测(predict)时，不生效，所有神经元均保留也就是不进行dropout。**

>  在Dropout的原始提出中，将剩下的神经元的输出按比例缩放是为了保持期望的输入值的总和不变，从而避免在训练过程中引入偏差。这个缩放操作有助于保持模型的输出的期望值与训练时相同，从而减少模型训练过程中的抖动。
>
>  具体来说，假设一个神经元的输出为x，训练时的Dropout概率为p。在训练过程中，该神经元的输出x会被置为零的概率为p，保持原值的概率为1-p。为了保持期望的输入值总和不变，我们需要对保留的输出进行缩放。
>
>  假设原始的输出为x，被保留的输出（未置零）为x'。由于在训练过程中，被保留的输出的期望值为原始输出的期望值乘以保留概率(1-p)，因此可以通过对被保留的输出进行缩放，将其除以保留概率(1-p)来保持期望值不变。
>
>  这样做的好处是，在训练过程中，由于被置零的输出被清除掉了，而被保留的输出又按比例缩放，可以保持模型在训练和测试过程中输入值的总和大致相等。这有助于减少训练过程中的抖动，使得模型更加稳定。
>
>  需要注意的是，在实际应用中，为了简化计算，有些实现并不对被保留的输出进行缩放，而是在测试阶段将所有神经元的输出都乘以保留概率(1-p)。这种近似的方法在实际中也取得了良好的效果。
>

kears 的函数有一个参数，即`rate`，表示要将输入张量的多少比例的元素设置为0。例如，如果`rate=0.2`，则在训练期间，`layers.Dropout`会在每次前向传播时随机地将20%的神经元输出设置为0。在测试期间，`layers.Dropout`的行为与正常的前向传播相同。

`layers.Dropout`的实现原理是，在训练期间，它会随机地丢弃一些神经元，使得**网络不会过于依赖某些特定的神经元**，从而减少过拟合。在测试期间，所有神经元都会被保留，因此网络可以以全力运行，得到更好的预测结果。

需要注意的是，`layers.Dropout`**通常应该在激活函数之前使用**，这样才能在不丢失任何信息的情况下减少过拟合。同时，`rate`的取值需要根据实际情况来确定，**通常在0.2到0.5之间**。

```python
Drop1 = layers.Dropout(0.2)(Con1) # 0.2就表示随机丢掉20%的神经元不激活。
```

https://blog.csdn.net/weixin_43290383/article/details/121601340

### GlobalPooling

>  将特征图所有像素值相加求平局，得到一个数值，**即用该数值表示对应特征图**。

在上文所说的Dropout有效避免过拟合后还有一种方案降低过拟合. 

**目的**：替代全连接层

**效果**：**减少参数数量，减少计算量，减少过拟合**

如下图所示。假设最终分成10类，则最后卷积层应该包含10个滤波器（即输出10个特征图），然后按照全局池化平均定义，分别对每个特征图，累加所有像素值并求平均，最后得到10个数值，将这10个数值输入到softmax层中，得到10个概率值，即这张图片属于每个类别的概率值。

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/c3aa9f461617403b9f92b22ba61c0632.png" alt="在这里插入图片描述" style="zoom: 33%;" />

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/971a2e67797e44c1a0c1930c00e08cad.png" alt="img" style="zoom:67%;" />

传统来说我们需要Flat数据后通过全连接层表达特征输出概率（这种结构将卷积结构与传统的神经网络分类器联系起来。它将**卷积层视为特征提取器**，并以传统方式对所得特征进行分类。），现在替换成GlobalPool输出概率，这里的特征表示部分则全部放在了最后一层的卷积层，这可能会有助于减小模型体积，提高表示能力。

如果要预测K个类别，在卷积特征抽取部分的最后一层卷积层，就会生成K个特征图，然后通过全局平均池化就可以得到 K个1×1的特征图，将这些1×1的特征图输入到softmax layer之后，每一个输出结果代表着这K个类别的概率（或置信度 confidence），起到**取代全连接层**的效果。

> 1. 和全连接层相比，使用全局平均池化技术，**强制特征图和类别之间**的对应关系，原生于卷积结构，是一种更**朴素**的卷积结构选择。因此，特征图可以很容易地解释为类别置信度图。
> 2. 全局平均池化层**不需要参数**，避免在该层产生过拟合。
> 3. 全局平均池化**总结空间信息**，对输入的空间平移变化的鲁棒性更强。

论文参考：

`[Lin M, Chen Q, Yan S. Network in Network[J\]. arXiv preprint arXiv:1312.4400, 2013.]`(https://arxiv.org/pdf/1312.4400.pdf)

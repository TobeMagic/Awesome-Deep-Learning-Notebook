# 傅里叶变换原理

 **傅里叶变换原理是什么**

傅里叶变换是一种数学工具，它可以将一个信号（如时域信号）**表示为不同频率的正弦和余弦函数的叠加**。具体来说，傅里叶变换可以将时域信号转换为频域信号，这样我们就可以更好地理解信号的频率内容和振幅特征。

傅里叶变换的公式为：

$$X(f)=\int_{-\infty}^{\infty}x(t)e^{-j2\pi ft}dt$$

其中，$x(t)$ 表示时域信号，$X(f)$ 表示其对应的频域信号，$f$ 表示频率，$j$ 表示虚数单位。这个公式意味着，将时域信号 $x(t)$ 乘以一个指数函数 $e^{-j2\pi ft}$，然后在整个时间域上进行积分，可以得到对应的频域信号 $X(f)$。

需要注意的是，傅里叶变换**假设信号是连续的并且无限长的，因此在实际应用中需要对信号进行采样和截断**。此外，由于傅里叶变换输出的是一个连续的频域信号，通常使用离散傅里叶变换（DFT）或快速傅里叶变换（FFT）对其进行==离散==化和计算。

**傅里叶变换有多个性质，其中一些重要的性质如下：**

1. 线性性质：对信号进行加权和或线性组合，其傅里叶变换等于各自变换的加权和或线性组合。
2. 移位性质：对时间域信号进行时间上的平移，其傅里叶变换会发生**相位旋转**，即在频率轴上移动相应的位置。
3. 翻转性质：对时间域信号进行翻转，其傅里叶变换会发生频率轴上的翻转。
4. 周期性质：周期函数的傅里叶变换是离散的，仅在某些离散的频率上有能量。
5. ==对称性质==：实函数的傅里叶变换是复共轭对称的，即变换结果的虚部是镜像对称的。

这是因为对于实函数的傅里叶变换，可以看作是对一个偶函数和一个奇函数进行变换。偶函数在频率轴上是实对称的，即其变换结果的虚部为0；奇函数在频率轴上是虚对称的，即其变换结果的实部为0。因此，实函数的变换结果是复共轭对称的，其虚部是镜像对称的。

对于实数序列的傅里叶变换，因为傅里叶变换是对于无限长的序列进行的，所以需要对序列进行零填充（或补零）以达到要求。

## **使用`np.fft.rfft`还是`np.fft.fft`**

傅里叶变换是将一个信号在频域中进行表示，因此输出的结果是一组频谱系数。对于实数信号，其频域表示是==对称==的，因此在计算时可以只计算其非负频率部分（即正频率部分），而省略负频率部分，因为负频率部分和正频率部分是共轭对称的。

其次，对于一个离散信号，其傅里叶变换也是离散的，输出结果为离散频谱系数。根据采样定理，一个离散信号的最高频率分量不会超过采样频率的一半。因此，对于离散信号的傅里叶变换，其频域表示的长度最多为采样率的一半，即$N/2+1$。这就是为什么 `np`在傅里叶变换的计算过程中，一般使用复数表示信号的频谱。假设原始信号的长度为N，则经过FFT计算后得到的频谱数据也有N个值。

由于频域上的对称性，实际上只需要知道前一半频谱的信息，后一半的信息可以通过前一半得到。因此，输出数组的长度只需要是前一半的长度再加1即可。

具体来说，对于长度为N的信号，经过FFT计算后得到的频谱数据的前N/2 + 1个值对应的是频率从0到Nyquist频率（采样率的一半）的频率分量，而剩余的N/2-1个值则是由这些值得到的镜像对称的负频率分量。为了**避免重复计算和存储这些镜像对称的值，输出数组的长度只取前N/2+1个值即可**。

而FFT可以保留更加详细的信息，因为它可以将时间域上的信号转换到频率域上进行分析，从而提取出信号中的不同频率成分。FFT使用的是DFT算法，可以高效地计算离散频率的傅里叶变换，因此在实际应用中比DFT更加常用。

不过，与DFT相同，FFT在进行频域分析时也需要满足采样定理的要求，即采样频率必须高于信号最高频率的两倍以上，才能避免混叠误差的发生。此外，FFT也有一些局限性，比如在信号长度不足或者信号中存在噪声时，可能会导致分析结果不准确。因此，在实际应用中，需要根据具体情况选择合适的算法和参数。

> 傅里叶变换后的复数包含了两个部分：幅度和相位。其中幅度表示信号在不同频率上的强度，而相位则表示信号的不同频率分量之间的时间相对关系。
>
> 具体来说，傅里叶变换将一个时域上的信号分解成多个不同频率的正弦波组成的谱。这些正弦波在频域上的振幅就是该频率下的幅度，而它们之间的相位差则反映了不同频率分量之间的时间相对关系。
>
> 因此，傅里叶变换后的复数的相位信息可以告诉我们不同频率分量之间的时间相对关系。这可以用于许多应用中，例如音频处理、信号处理等。

# DDAE 和 DAE 去噪算法

DAE (Denoising AutoEncoder) 算法是一种无监督学习算法，它的主要目的是通过自编码器 (Autoencoder) 对输入数据进行特征提取和去噪处理。DAE 的网络结构通常包含一个编码器和一个解码器。**编码器将输入数据映射到一个低维的潜在空间表示，解码器则将低维表示还原回原始的高维数据**。在训练过程中，DAE 通过最小化重构误差来学习数据的特征表示和去噪能力。

具体而言，DAE 在训练时先对原始数据添加一定的噪声，然后将添加噪声的数据作为模型的输入，希望模型能够输出去除噪声的原始数据。在训练完成后，DAE 可以用于对新的噪声数据进行去噪处理。DAE 模型可以使用多种深度学习算法进行训练，包括基于反向传播的算法，如梯度下降、随机梯度下降、Adam 等。

DDAE (Denoising Deep Autoencoder) 是 DAE 的一种改进，它引入了多层网络结构，并通过逐层训练来学习数据的高级特征表示。DDAE 可以比 DAE 学习到更抽象的特征表示，并且在处理复杂数据时具有更好的表达能力和鲁棒性。DDAE 的训练过程也与 DAE 相似，只不过需要逐层训练多个自编码器。

相比于传统的 DAE 算法，DDAE 能够处理更加复杂的数据，例如图像、视频、语音等。在语音信号的去噪方面，DDAE 可以学习到更高阶的特征表示，从而提高去噪效果。但是，DDAE 模型的训练和计算成本通常会比 DAE 更高，因此需要更大量的训练数据和更强的计算能力。

## 代码模板

**DAE（Deep Autoencoder）**是一种常用于无监督学习的深度神经网络模型。下面是一个标准的DAE模板代码，以Python语言为例：

```PYTHON
pythonCopy Codeimport numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 构建编码器和解码器
def build_autoencoder(input_shape, code_size):
    input_layer = Input(shape=input_shape)
    code_layer = Dense(code_size, activation='relu')(input_layer)
    output_layer = Dense(input_shape[0], activation='sigmoid')(code_layer)
    
    encoder = Model(input_layer, code_layer)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

# 加载输入数据
def load_data():
    data = np.load('data.npy')   # 从文件中加载数据
    return data

# 训练模型
def train(autoencoder, X_train, epochs):
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=32)

# 使用模型进行特征提取
def encode(encoder, X):
    return encoder.predict(X)

# 使用模型进行重构
def reconstruct(autoencoder, X):
    return autoencoder.predict(X)
```

这个模板代码包括了DAE模型的构建、输入数据的加载和预处理、模型训练和使用等基本步骤

**DDAE（Denoising Deep Autoencoder）**是一种常用于信号去噪的深度学习模型。下面是一个标准的DDAE模板代码，以Python语言为例：


> DDAE模型通常需要多个数据才能进行训练，因为它是一种无监督学习模型，没有标签信息指导学习。
>
> 在DDAE模型中，我们需要准备两组数据：**干净的数据和带有噪声的数据**。这两组数据应该来自于同一个数据集，且数量要相等。其中，干净的数据用于作为模型的目标输出，带有噪声的数据则用于作为模型的输入。模型的目标是尽可能地将带有噪声的数据还原成干净的数据。
>
> 在训练过程中，我们使用带有噪声的数据作为输入，干净的数据作为目标输出。通过反向传播算法，模型会不断更新权重和偏置，最小化输入数据与目标输出之间的均方误差（Mean Squared Error, MSE）。
>
> 需要注意的是，DDAE模型的性能受到训练数据的质量和数量的影响。如果训练数据量过少，模型可能会过拟合；如果训练数据质量差，模型可能会出现欠拟合。因此，在实际应用中，需要仔细选择适当的训练数据集。

```python
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 构建编码器和解码器
def build_autoencoder(input_shape, code_size):
    input_layer = Input(shape=input_shape)
    code_layer = Dense(code_size, activation='relu')(input_layer)
    output_layer = Dense(input_shape[0], activation='sigmoid')(code_layer)
    
    encoder = Model(input_layer, code_layer)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

# 加载输入数据，并添加噪声
def load_data():
    data = np.load('data.npy')   # 从文件中加载数据
    noisy_data = data + 0.1 * np.random.randn(*data.shape)   # 添加高斯噪声
    return data, noisy_data

# 训练模型
def train(autoencoder, X_train, X_noisy, epochs):
    autoencoder.fit(X_noisy, X_train, epochs=epochs, batch_size=32)

# 使用模型进行信号去噪
def denoise_signal(encoder, X_noisy):
    return encoder.predict(X_noisy)
```

这个模板代码包括了DDAE模型的构建、输入数据的加载和预处理、模型训练和使用等基本步骤。

需要注意的是，这只是一个简单的模板代码，实际应用中可能需要根据具体的问题进行修改和扩展。例如，模型结构的调整、超参数的设置、数据增强等都可能对模型的性能产生影响。



# 自监督学习

## 音频领域对比学习

### **wav2vec系列**

wav2vec系列工作由facebook AI Research团队提出，包括wav2vec、vq-wav2vec、wav2vec2.0，效仿nlp上的word2vec，是语音的一种通用特征提取器。建议先了解奠基之作CPC论文。

#### wav2vec

论文：[wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/pdf/1904.05862.pdf)

本文提出一种无监督的语音预训练模型 wav2vec，可迁移到语音下游任务。模型预训练一个简单的多层卷积神经网络，并提出了一种**噪声对比学习**二分类任务(noise contrastive binary classification task)，从而使得wav2vec可以在大量未标注的数据上进行训练。实验结果表明wav2vec预训练得到的speech representation超越了帧级别的音素分类任务并且可以显著提升ASR模型的表现，同时，**完全卷积架构**与使用的递归模型相比，可以在硬件上并行计算。

模型结构如下图，首先将原始音频x编码为潜在空间z的 encoder network（5层卷积），再将潜在空间z转换为contextualized representation（9层卷积），最终特征维度为512x帧数。目标是在特征层面使用当前帧预测未来帧。

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/wav2vec.png" alt="img" style="zoom:50%;" />

模型将原始音频信号 x 作为输入，基于历史信息和当前输入的信息预测未来的某些采样点，这里使用了两个编码器进行计算。

- 编码器网络f(encoder network) 将音频信号嵌入到特征空间(latent space) 中将每个xi映射为一个特征向量zi, 类似于language model模型那样获得一个编码向量, 再基于此预测某个zi, 这里j>i;
- 上下文网络g(context network) 结合了多个时间步长编码器以获得上下文表示(contextualized representations) 如图1。将多个zi转化为context representation C.这里有 $ c_ {i} $ =g( $ z_ {i} $ , $ z_ {i-1} $ $ \cdots $ $ z_ {v} $ )。这里的v为感受野(receptive field size)

然后, 两个网络的输出Z, C都用于损失函数(loss function) 的计算。作者在实验中使用了两种不同的感受野模型, 一种为普通规模, 用来在一般数据集上训练, 另一种则是大规模(wav2vec larqe) 用来在大数据集上训练。在这两种模型中的感受野分别对应210ms和810ms.

模型的loss中自然要包含预测未来某个z的损失。然而仅仅有正例是不够的, 因此作者利用了负采样技术, 作者从一个概率分布 $ p_ {n} $ 中采样出负样本z,最终模型的loss为区分正例和反例的contrastive loss :

> 这里的损失函数在CPC论文的基础上继续改进，不再使用分母的形式避免除0之外，添加了配置参数λ 设置负样本的个数而不是计算全部减小了模型大小和计算量，并使用sigmoid将数据缩放在0-1

 $ L_ {k} $ =- $ Z_ {i=1}^ {T-k} $ ( $ \log $ $ \sigma $ ( $ z^ {T}_ {i+k} $ $ h_ {k} $ ( $ c_ {i} $ ))+ $ \lambda $E[ $ \log $ $ \sigma $ ( $- \widetilde{z}^{T}h_ {k} $ ( $ c_ {i} $ ))]

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/v2-f68e4ab7c537b660024c2e992bd1c51f_720w.webp" alt="img" style="zoom: 33%;" />

对于正样本，损失函数的第一项是负对数似然损失。它衡量了模型预测下一个上下文的编码的准确性。具体地说，对于每个上下文$c_i$，模型使用当前上下文的编码作为输入，然后预测下一个上下文的编码。通过比较预测的编码和实际编码，我们可以计算出负对数似然损失。该损失项的表示为$Z_{i=1}^{T-k}\log\sigma(z^{T}_{i+k}h_{k}(c_{i}))$，其中$Z_{i=1}^{T-k}$是对所有上下文的求和，$z^{T}_{i+k}$是下一个上下文的实际编码，$h_{k}(c_{i})$是模型对当前上下文的预测编码，$\sigma$是sigmoid函数，将编码二者相似度转换为概率。

对于负样本，损失函数的第二项是对预测的负编码的正则化项。这个负编码是通过对当前上下文的预测编码$h_{k}(c_{i})$与一个随机生成的编码$\widetilde{z}^{T}$的点积得到的。通过对负编码的正则化，我们鼓励模型不仅仅关注正确的预测，还要确保预测的编码与随机编码之间的点积尽可能小。这个正则化项的表示为$\lambda E[\log\sigma(-\widetilde{z}^{T}h_{k}(c_{i}))]$，其中$\lambda$是正则化的权重，$E$是对随机编码的期望。

通过将这两个项相加，我们得到了wav2vec模型的总损失函数。这个损失函数的目标是最小化正样本的负对数似然损失，同时确保负样本的正则化项尽可能小。这样，模型可以学习到一个有效的编码器，将语音信号映射到有用的表示空间中，以便后续的语音识别任务。

#### vq-wav2vec

论文：[vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/pdf/1910.05453v1.pdf)

本文基于wav2vec，将连续特征z通过提出的量化模块，变成离散特征z‘，实现特征空间从无限的连续到有限的离散的转换过程。这作为一个创新点，也是它有效的重要原因之一。

乘积量化模块的作用是将Encoder的输出离散化成为了一组数量有限的语音表示，对于乘积量化的解释如下

```
乘积量化，是指笛卡尔积（Cartesian product），意思是指把原来的向量空间分解为若干个低维向量空间的笛卡尔积，并对分解得到的	低维向量空间分别做量化（quantization）。这样每个向量就能由多个低维空间的量化code组合表示。这里的量化不是将float量化成int，而是把连续空间量化成有限空间。

1、乘积量化的原理

说人话，就是把原来连续的特征空间假设是d维，拆分成G个子空间（codebook），每个子空间维度是d/G。然后分别在每个子空间里面聚类（比如K-mean算法），一共获得V个中心和其中心特征。每个类别的特征用其中心特征代替。

结果就是，原来d维的连续空间（有无限种特征表达形式），坍缩成了有限离线的空间[GxV]，其可能的特征种类数就只有G*V个。

2、乘积量化巧妙在哪儿

乘积量化操作通过将无限的特征表达空间坍缩成有限的离散空间，让特征的鲁棒性更强，不会受少量扰动的影响（只要还在某一类里面，特征都由中心特征来代替）。这个聚类过程也是一个特征提取的过程，让特征的表征能力更强了。
```

文中提出了两种量化方法，Gumbel softmax和K-Means，如下图。 其中，左右两个部分中的 e1 … ev，就是码本（记录特征集，可以理解为 BERT 中的词表），Gumbel通过逻辑值最大化（回传时使用Gumbel softmax来保证可导）找对应码本条，K-Means通过计算与码本距离来找最小距离的码本条。

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/2021041709351933.png" alt="[]" style="zoom: 33%;" />

#### wav2vec2

本文基于wav2vec，结合了vq-wav2vec的量化模块和Transformer，提出了wav2vec2.0，如下图。其中，**模型由CNN和 Transformer 构成**，可以分为**特征编码器模块、上下文编码器模块和量化模块**三部分。特征编码器模块由卷积神经网络构成，把原始音频信号波形转化为隐层语音表征；量化模块把隐层语音表征转化为量化表征，作为对比目标；上下文编码器模块把隐层语音表征转化为上下文表征。上下文表征和量化表征通过对比任务实现自监督预训练。

**论文：[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477v1.pdf)**

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/image-20240126210610133.png" alt="image-20240126210610133" style="zoom:50%;" />

训练好的 wav2vec2.0 模型可以看作一个特征提取器，声音信号通过 wav2vec2.0 模型输出一个表征向量，该表征向量可以代替传统的 MFCC、频谱等声学特征，也可以采用特征融合的方式与传统声学特征一起使用，以充分利用不同特征间的互补性。

> 预训练-微调模式继CV与NLP之后开始席卷语音领域，在实践中也印证了wav2vec2.0在小规模数据集上fine-tune之后确实能达到非常好的效果。但凡事都是有代价的，正如预训练模型给CV、NLP带去的困扰那样，wav2vec2.0虽然效果喜人，但**无奈太过笨重**，要实现线上使用还需要进行一定的压缩与加速。总的来说，瑕不掩瑜，wav2vec2.0依然是一个非常适合**低资源冷启动项目**的基础模型。

模型的整体结构如下图，以下具体讲解结构。

![img](https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/20210417093437857.png)

通过多层卷积神经网络对语音音频进行编码，然后屏蔽生成的潜在语音表示的范围，类似于屏蔽语言建模潜在表示被馈送到 Transformer 网络以构建上下文表示，并通过对比任务训练模型，其中真正的潜在表示将与干扰项区分开来 。作为训练的一部分，我们通过gumbel softmax学习离散语音单元来表示对比任务中的潜在表示，我们发现这比非量化目标更有效。对未标记语音进行预训练后，对模型进行微调具有连接主义时间分类 (CTC) loss 的标记数据，用于下游语音识别任务 

#####encoder network

wav2vec2.0的encoder network由7层卷积层构成，结构如下图所示

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/FeatureEncoder.png" alt="img" style="zoom:50%;" />

文章使用了7层的CNN，步长分别为(5,2,2,2,2,2,2)，卷积核宽度为(10,3,3,3,3,2,2)，假设输入语音的长度为(1,x)：
    cnn0 (x-10)/5+1=x/5-1
    cnn1 ((x/5-1)-3)/2+1=x/10-1
    cnn2 x/20-1
    cnn3 x/40-1
    cnn4 x/80-1
    cnn5 x/160
    cnn6 x/320
采样率为16k的情况下，1s的语音长度对应矩阵(1,16000)，论文中的channels大小设置的为512，对应的输出为(512,16000/320)=(512,50)，可以得到50个512维的向量，相当于每20ms产生一个512维的特征向量。以下是encoder network模型源码输出部分，将输出z 潜在表示.

```
  (feature_extractor): ConvFeatureExtractionModel(
    (conv_layers): ModuleList(
      (0): Sequential(
        (0): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (3): ReLU()
      )
      (1): Sequential(
        (0): Conv1d(512, 512, kernel_size=(8,), stride=(4,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (3): ReLU()
      )
      (2-4): 3 x Sequential(
        (0): Conv1d(512, 512, kernel_size=(4,), stride=(2,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (3): ReLU()
      )
      (5-6): 2 x Sequential(
        (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
        (1): Dropout(p=0.0, inplace=False)
        (2): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (3): ReLU()
      )
    )
  )
```
##### context network

整体结构图中的context包括左右两部分，左边负责将z转换成c（对应wav2vec2特征），右边负责将z离散化以计算损失。Encoder由Transformer组成，base版本的tfm层数为12，large版为24。

左边部分中，对于输入512x50的z，有：
post_extract_proj: 768x50
apply_mask->pos_conv->LN: 768x50
Transformer*12: 768x50
choose_masking: 768xM，M为mask的帧数
final_proj: 256xM

右边部分中，对于输入512x50的z，有：
choose_masking: 512xM
quantizer: 256xM
project_q: 256xM

其中，量化的参数有：码本个数G=2，每个码本的条目个数V=320，条目的维度d/G=256/2=128。参数含义：G=latent_groups，V=latent_vars，d=vq_dim。
具体的quantizer流程如下图所示，前向的时候直接找出来最大值对应的码本中的条目，相当于是一个离散的操作，但是这个步骤不可导，无法进行反向传播，为了解决这个问题，采用了gumbel softmax操作。

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/quantizer.png" alt="img" style="zoom: 33%;" />

**这里需要注意的是PositionEmbedding**，wav2vec使用一个卷积层来作为PE，并将PE加到hidden_state中后传入Transformers。wav2vec2原文描述为：

```
Instead of fixed positional embeddings which encode absolute positional information, we use a convolutional layer similar to which acts as relative positional embedding.
我们使用类似于相对位置嵌入的卷积层，而不是编码绝对位置信息的固定位置嵌入。
```

利用卷积PE替代传统的三角函数PE的做法自FAIR的另一个研究成果——[《**Transformers with convolutional context for ASR**》](https://arxiv.org/abs/1904.11660)。在这篇文章中，作者通过实验比对了多个PE的效果，最终卷积PE脱颖而出，被wav2vec2继承了下来。

###### **Mask（遮罩）**

一帧被选为mask区域起点的概率p是0.065，mask长度M为10（文章的表述方法），对应的训练参数为–mask-length 10 --mask-prob 0.65

获取mask区域的个数num_mask
num_mask=语音长度/mask_length*mask_prob，由于存在overlap，所以最终mask的区域会少
mask lengths有四种计算方式：
static
uniform
normal
poisson
随机选取num_mask的起点，做mask，mask使用的向量
torch.FloatTensor(args.encoder_embed_dim).uniform_()
最终的效果大概有49%的帧会做mask，平均mask span的长度为14.7帧

###### **损失函数**

wav2vec 2.0的损失函数由两部分构成，对抗性损失Lm和多样性损失Ld。

Lm的形式和CPC是相似的，区别在于使用余弦距离sim代替原来的linear映射层，同时用乘积量化的结果qt代替原来zt（表征能力更强嘛）。

Ld是新引入的多样性损失，其目的是监督乘积量化中的聚类过程，期望每个中心点尽量的远。其中G是codebook数量，V是聚类中心的数量，p是某个特征在某个(g,v)子空间的概率值，其具体表达式就是gumble softmax的表达式。

潜在表示到最终上下文表示模型源码如下(聚合器):

```
 (feature_aggregator): ConvAggegator(
    (conv_layers): Sequential(
      (0): Sequential(
        (0): ReplicationPad1d((1, 0))
        (1): Conv1d(512, 512, kernel_size=(2,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (1): Sequential(
        (0): ReplicationPad1d((2, 0))
        (1): Conv1d(512, 512, kernel_size=(3,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (2): Sequential(
        (0): ReplicationPad1d((3, 0))
        (1): Conv1d(512, 512, kernel_size=(4,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (3): Sequential(
        (0): ReplicationPad1d((4, 0))
        (1): Conv1d(512, 512, kernel_size=(5,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (4): Sequential(
        (0): ReplicationPad1d((5, 0))
        (1): Conv1d(512, 512, kernel_size=(6,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (5): Sequential(
        (0): ReplicationPad1d((6, 0))
        (1): Conv1d(512, 512, kernel_size=(7,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (6): Sequential(
        (0): ReplicationPad1d((7, 0))
        (1): Conv1d(512, 512, kernel_size=(8,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (7): Sequential(
        (0): ReplicationPad1d((8, 0))
        (1): Conv1d(512, 512, kernel_size=(9,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (8): Sequential(
        (0): ReplicationPad1d((9, 0))
        (1): Conv1d(512, 512, kernel_size=(10,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (9): Sequential(
        (0): ReplicationPad1d((10, 0))
        (1): Conv1d(512, 512, kernel_size=(11,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (10): Sequential(
        (0): ReplicationPad1d((11, 0))
        (1): Conv1d(512, 512, kernel_size=(12,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
      (11): Sequential(
        (0): ReplicationPad1d((12, 0))
        (1): Conv1d(512, 512, kernel_size=(13,), stride=(1,))
        (2): Dropout(p=0.0, inplace=False)
        (3): Fp32GroupNorm(1, 512, eps=1e-05, affine=True)
        (4): ReLU()
      )
    )
    (residual_proj): ModuleList(
      (0-11): 12 x None
    )
  )
  (wav2vec_predictions): Wav2VecPredictionsModel(
    (project_to_steps): ConvTranspose2d(512, 512, kernel_size=(1, 12), stride=(1, 1))
    (dropout): Dropout(p=0.0, inplace=False)
  )
  (dropout_feats): Dropout(p=0.0, inplace=False)
  (dropout_agg): Dropout(p=0.0, inplace=False)
)
```

几点说明：

1、一维卷积获得的结果z，一方面经过mask后直接送入到了transformer中，另一方面通过乘积量化的操作获得q，参与后面的损失函数。

2、其中ct，qt均是来源于mask的部分，非mask的部分是用于预测mask部分的。所以Lm中的样本组成是：transformer在第t个mask中心的输出ct——预测值，通过乘积量化在第t个mask中得到的聚类中心值qt——正样本（理论上完成聚类后，mask里面的特征因为相似，其都属于同一个类，所以就一个中心），从其他mask里面随机抽取的乘积量化结果q——负样本。

3、算法微调是在transformer后面接了一个linear层进行微调。

4、wav2vec 2.0中没有对ct再次进行映射，而是直接求ct和qt的相似度。如果网络训练得很好，ct跟qt非常相似，微调的时候，是不是就可以不用transformer了，直接使用qt来替换？有兴趣的小伙伴可以思考一下，为什么还是要用transformer?

**wav2vec3.0？**

现在wav2vec 2.0虽然效果很好，但还是由很多缺点。

1、模型结构中没有语言模型，需要外挂语言模型才能进行识别，不能做到端到端。

2、transformer结构只能做纯离线识别，没办法流式解码。

3、模型太大了，特别是预训练模型，非常消耗算力，普通人就不要幻想从零开始训练了，对预训练模型的依赖很高。

那么facebook会不会在未来推出一个3.0呢？把rnnt-t loss包含进来，然后在transformer中引入chunk机制，然后再搞一个涵盖各个语言、各个方言的超级预训练模型。这样对于任意下游任务，只需要通过少量数据，就可以实现一个高精度、纯端到端、流式解码的语音识别系统了？

最后，我在其他数据集上试了试wav2vec 2.0，效果确实很好，超过了精心训练的deepspeech、transformer等模型。所以强烈建议感兴趣的同学深入研究一下这篇论文。由于篇幅有限，很多算法细节都没有展开，以后希望有精力来把这个坑填了。

##### wav2vec2.0的使用

transformers库

```python
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")   # 用于ASR等，32维

audio_input, sample_rate = sf.read(path_audio)  # (31129,)
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values  # torch.Size([1, 31129])

logits = model(input_values).logits     # torch.Size([1, 97, 32])
predicted_ids = torch.argmax(logits, dim=-1)    # torch.Size([1, 97])

transcription = processor.decode(predicted_ids[0])  # ASR的解码结果

from transformers import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")    # 用于提取通用特征，768维
wav2vec2 = model(input_values)['last_hidden_state']     # torch.Size([1, 97, 768])，模型出来是一个BaseModelOutput的结构体。
```

或者 fairseq 源码库

```python
import torch
import fairseq

cp_path = 'wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
```

参考文章：
http://nelslip.ustc.edu.cn/2022/0509/c26914a562917/page.htm
https://qinyuenlp.com/article/1837c5011ace/
https://blog.csdn.net/xmdxcsj/article/details/115787729
https://zhuanlan.zhihu.com/p/390545403
https://tobefans.github.io/2021/11/24/wav2vec2/

**相关资源**

arXiv：https://arxiv.org/abs/2006.11477
GitHub(FAIR)：https://github.com/pytorch/fairseq
Github(HuggingFace)：https://github.com/huggingface/transformers/tree/main/src/transformers/models/wav2vec2
代码复现地址：
https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md
https://paperswithcode.com/paper/unsupervised-speech-recognition#code




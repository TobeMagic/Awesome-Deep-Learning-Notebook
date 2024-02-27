## 简介

官网：https://pytorch.org/
中文学习文档:https://pytorch.panchuang.net/ 

Torch是一个有大量机器学习算法支持的科学计算框架，是一个与Numpy类似的张量（Tensor） 操作库，其特点是特别灵活，但因其采用了小众的编程语言是Lua，所以流行度不高，这也就有了PyTorch的出现。所以其实Torch是 PyTorch的前身，它们的底层语言相同，只是使用了不同的上层包装语言。

> 现今任何一个深度学习框架都有其缺点，PyTorch也不例外，对比TensorFlow，其全面性处于劣势，目前PyTorch还不支持快速傅里 叶、沿维翻转张量和检查无穷与非数值张量；针对移动端、嵌入式部署以及高性能服务器端的部署其性能表现有待提升；其次因为这个框 架较新，使得他的社区没有那么强大，在文档方面其C库大多数没有文档。

安装方式在官网可提供（GPU的包会很大，CPU相对小很多，默认没有index-url就是CPU下载）

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/image-20240127112116694.png" alt="image-20240127112116694" style="zoom: 50%;" />

### pytorch结构图

![在这里插入图片描述](https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/89ef0eabf48a4c31b7403e26d1b5be77.png)

> 结构主要强调的是模块与模块之间的包含关系，而架构更多的强调模块之间的依赖关系

**Tensor**：整个torch操作的大部分数据类型都是Tensor，所以torch.Tensor呢可以说是整个torch的结构基础
**Storage**：Storage则是torch的存储基础，这个包管理着数据要怎么存放在硬件里面，比如是以bite的形式存储啊还是以char的形式存储，是存储在cpu中还是gpu中等等。
**autograd**：顾名思义，负责求导。
**optim**：顾名思义，负责优化网络参数，使得网络能够一步一步拟合所需要的函数
**cuda**：使用GPU加速计算的重要库
**utils**：可以帮助我们更好的训练网络。它下面的Data子包包括了Dataset模块，可以让我们简单的调用数据集，Dataloader则可以让我们方便的使用数据集。

nn这一个大模块负责搭建网络，它下面又有很多子模块。

> Parameter负责管理网络中的参数，比如是否需要求导
> init负责各种参数的初始化
> functional则包含了很多函数，比如激活函数ReLU、sigmoid，之类的。Nomalizations则是归一化函数，LossFunctions就是均方误差、交叉熵之类的损失函数了
> Containers是nn的主体，用来存放各种上面或者下面的结构的。用过pytroch的同学就知道，如果你要写一个网络模型，用类的方式来写，就得继承这个Module，相当于让你的网络类也具有容器的功能，可以往里放各种结构，比如全连接层，激活函数，卷积层之类的。sequential是更轻量化一点的容器，不过效果跟Module差不多。
> 然后就是最核心的网络层了，nn里面也集成了很多常用的网络层，比如全连接层Linear，然后一二三维的卷积层、或者一二三维的池化层之类的。
> nn里面也给我们直接集成了现在常用的网络模型，比如RNN、LSTM等，方便我们直接调用。

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/324935ad62934fb890e2560f270dc6be.png" alt="在这里插入图片描述" style="zoom:50%;" />

### 架构图

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/26ede6729d8f4bbaa2760a66593a1260.jpg" alt="在这里插入图片描述" style="zoom:50%;" />

参考文章：https://blog.csdn.net/qq_28726979/article/details/120690343

## 模型保存与加载

当保存和加载模型时，需要熟悉三个核心功能：

1. `torch.save`：将序列化对象保存到磁盘。此函数使用Python的`pickle`模块进行序列化。使用此函数可以保存如模型、tensor、字典等各种对象。
2. `torch.load`：使用pickle的`unpickling`功能将pickle对象文件反序列化到内存。此功能还可以有助于设备加载数据。
3. `torch.nn.Module.load_state_dict`：使用反序列化函数 state_dict 来加载模型的参数字典。

### 状态字典：state_dict

在PyTorch中，`torch.nn.Module`模型的可学习参数（即权重和偏差）包含在模型的参数中，（使用`model.parameters()`可以进行访问）。 `state_dict`是Python字典对象，它将每一层映射到其参数张量。注意，只有具有可学习参数的层（如卷积层，线性层等）的模型 才具有`state_dict`这一项。目标优化`torch.optim`也有`state_dict`属性，它包含有关优化器的状态信息，以及使用的超参数。

因为state_dict的对象是Python字典，所以它们可以很容易的保存、更新、修改和恢复，为PyTorch模型和优化器添加了大量模块。

下面通过从简单模型训练一个[分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)中来了解一下`state_dict`的使用。

```
# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = TheModelClass()

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

- 输出

```
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```

### 加载`state_dict`（推荐使用）

- 保存

```
torch.save(model.state_dict(), PATH)
```

- 加载

```
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

当保存好模型用来推断的时候，只需要保存模型学习到的参数，使用`torch.save()`函数来保存模型`state_dict`,它会给模型恢复提供 **最大的灵活性**，这就是为什么要推荐它来保存的原因。

在 PyTorch 中最常见的模型保存使‘.pt’或者是‘.pth’作为模型文件扩展名。

请记住，在运行推理之前，务必调用`model.eval()`去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致 模型推断结果不一致。

- 注意

`load_state_dict()`函数只接受字典对象，而不是保存对象的路径。这就意味着在你传给`load_state_dict()`函数之前，你必须反序列化 你保存的`state_dict`。例如，你无法通过 `model.load_state_dict(PATH)`来加载模型。

### 保存/加载完整模型

- 保存

```
torch.save(model, PATH)
```

- 加载

```
# 模型类必须在此之前被定义
model = torch.load(PATH)
model.eval()
```

此部分保存/加载过程使用最直观的语法并涉及**最少量的代码**。以 Python `pickle 模块的方式来保存模型。这种方法的缺点是序列化数据受 限于某种特殊的类而且需要确切的字典结构。这是因为pickle无法保存模型类本身。相反，它保存包含类的文件的路径，该文件在加载时使用。 因此，当在其他项目使用或者重构之后，您的代码可能会以各种方式中断。

在 PyTorch 中最常见的模型保存使用‘.pt’或者是‘.pth’作为模型文件扩展名。

请记住，在运行推理之前，务必调用`model.eval()`设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致模型推断结果不一致。

####保存和加载 Checkpoint 用于推理/继续训练

- 保存

```
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```

- 加载

```
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

当保存成 Checkpoint 的时候，可用于推理或者是继续训练，保存的不仅仅是模型的 state_dict 。保存优化器的 state_dict 也很重要, 因为它包含作为模型训练更新的缓冲区和参数。你也许想保存其他项目，比如最新记录的训练损失，外部的`torch.nn.Embedding`层等等。

要保存多个组件，请在字典中组织它们并使用`torch.save()`来序列化字典。PyTorch 中常见的保存checkpoint 是使用 .tar 文件扩展名。

要加载项目，首先需要初始化模型和优化器，然后使用`torch.load()`来加载本地字典。这里,你可以非常容易的通过简单查询字典来访问你所保存的项目。

请记住在运行推理之前，务必调用`model.eval()`去设置 dropout 和 batch normalization 为评估。如果不这样做，有可能得到不一致的推断结果。 如果你想要恢复训练，请调用`model.train()`以确保这些层处于训练模式。

#### 在一个文件中保存多个模型

- 保存

```
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)
```

- 加载

```
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```

当保存一个模型由多个`torch.nn.Modules`组成时，例如GAN(对抗生成网络)、sequence-to-sequence (序列到序列模型), 或者是多个模 型融合, 可以采用与保存常规检查点相同的方法。换句话说，保存每个模型的 state_dict 的字典和相对应的优化器。如前所述，可以通 过简单地将它们附加到字典的方式来保存任何其他项目，这样有助于恢复训练。

PyTorch 中常见的保存 checkpoint 是使用 .tar 文件扩展名。

要加载项目，首先需要初始化模型和优化器，然后使用`torch.load()`来加载本地字典。这里，你可以非常容易的通过简单查询字典来访问你所保存的项目。

请记住在运行推理之前，务必调用`model.eval()`去设置 dropout 和 batch normalization 为评估。如果不这样做，有可能得到不一致的推断结果。 如果你想要恢复训练，请调用`model.train()`以确保这些层处于训练模式。

#### 使用在不同模型参数下的热启动模式

- 保存

```
torch.save(modelA.state_dict(), PATH)
```

- 加载

```
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

在迁移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见的情况。利用训练好的参数，有助于热启动训练过程，并希望帮助你的模型比从头开始训练能够更快地收敛。

无论是从缺少某些键的 state_dict 加载还是从键的数目多于加载模型的 state_dict , 都可以通过在`load_state_dict()`函数中将`strict`参数设置为 False 来忽略非匹配键的函数。

如果要将参数从一个层加载到另一个层，但是某些键不匹配，主要修改正在加载的 state_dict 中的参数键的名称以匹配要在加载到模型中的键即可。

#### 通过设备保存/加载模型

##### 保存到 CPU、加载到 CPU

- 保存

```
torch.save(model.state_dict(), PATH)
```

- 加载

```
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

当从CPU上加载模型在GPU上训练时, 将`torch.device('cpu')`传递给`torch.load()`函数中的`map_location`参数.在这种情况下，使用 `map_location`参数将张量下的存储器动态的重新映射到CPU设备。

##### 保存到 GPU、加载到 GPU

- 保存

```
torch.save(model.state_dict(), PATH)
```

- 加载

```
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# 确保在你提供给模型的任何输入张量上调用input = input.to(device)
```

当在GPU上训练并把模型保存在GPU，只需要使用`model.to(torch.device('cuda'))`，将初始化的 model 转换为 CUDA 优化模型。另外，请 务必在所有模型输入上使用`.to(torch.device('cuda'))`函数来为模型准备数据。请注意，调用`my_tensor.to(device)`会在GPU上返回`my_tensor`的副本。 因此，请记住手动覆盖张量：`my_tensor= my_tensor.to(torch.device('cuda'))`。

##### 保存到 CPU，加载到 GPU

- 保存

```
torch.save(model.state_dict(), PATH)
```

- 加载

```
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# 确保在你提供给模型的任何输入张量上调用input = input.to(device)
```

在CPU上训练好并保存的模型加载到GPU时，将`torch.load()`函数中的`map_location`参数设置为`cuda:device_id`。这会将模型加载到 指定的GPU设备。接下来，请务必调用`model.to(torch.device('cuda'))`将模型的参数张量转换为 CUDA 张量。最后，确保在所有模型输入上使用 `.to(torch.device('cuda'))`函数来为CUDA优化模型。请注意，调用`my_tensor.to(device)`会在GPU上返回`my_tensor`的新副本。它不会覆盖`my_tensor`。 因此， 请手动覆盖张量`my_tensor = my_tensor.to(torch.device('cuda'))`。

##### 保存 `torch.nn.DataParallel` 模型

- 保存

```
torch.save(model.module.state_dict(), PATH)
```

- 加载

```
# 加载任何你想要的设备
```

`torch.nn.DataParallel`是一个模型封装，支持并行GPU使用。要普通保存 DataParallel 模型, 请保存`model.module.state_dict()`。 这样，你就可以非常灵活地以任何方式加载模型到你想要的设备中。
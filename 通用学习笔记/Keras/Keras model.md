## 加载模型

在 `TensorFlow `中，可以使用 `model.save()` 方法将模型保存为 Keras HDF5 格式的模型文件。保存模型后，可以使用以下代码来查看模型结构和参数：

```python
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('model.h5')

# 查看模型结构
model.summary()

# 查看模型参数
print(model.get_weights())
```

在上面的代码中，`load_model('model.h5')` 用于加载保存的模型文件。`model.summary()` 用于打印模型的结构，包括每一层的名称、输出形状和参数数量等信息。`model.get_weights()` 用于打印模型的参数，包括每一层的权重和偏置等。

## 加载参数

在`TensorFlow`中，使用`Keras` API训练的模型可以通过保存为HDF5格式的权重文件保存和加载。如下所示，是使用`model.save_weights()`方法将模型权重保存到HDF5文件：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 定义模型结构
model = Sequential([
    Dense(64, input_shape=(784,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# 训练模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# 保存模型权重到HDF5文件
model.save_weights('model_weights.h5')
```

在保存模型权重后，可以通过以下代码来加载模型权重并进行预测：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 定义模型结构
model = Sequential([
    Dense(64, input_shape=(784,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# 加载模型权重
model.load_weights('model_weights.h5')

# 进行预测
predictions = model.predict(test_images)
```

在上面的代码中，`model.load_weights('model_weights.h5')`用于加载保存的HDF5权重文件。加载权重后，可以使用`model.predict()`方法进行预测操作，返回模型的预测结果。需要注意的是，**加载模型权重时需要保证模型结构与训练模型时的结构完全一致**，否则会导致加载失败。

## 模型保存

https://blog.csdn.net/zhangpeterx/article/details/90897439

模型保存

```python
# 将模型保存到Tensorflow SavedModel或单个HDF5文件。
model.save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None)

```

```python
model.save_weights()# 保存所有层权重
save_weights(self, filepath, overwrite=True, save_format=None)
```

## Backend_lists

https://blog.csdn.net/yunfeather/article/details/106461754

**model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准**

callbacks_list

一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合`keras.callbacks.EarlyStopping`就是用来提前结束训练的。

```python
"""
tensorflow.keras.callbacks.EarlyStopping :
monitor: 被监测的数据。
min_delta: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
patience: 没有进步的训练轮数，在这之后训练就会被停止。
verbose: 详细信息模式。
mode: {auto, min, max} 其中之一。 在 min 模式中， 当被监测的数据停止下降，训练就会停止；在 max 模式中，当被监测的数据停止上升，训练就会停止；在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
baseline: 要监控的数量的基准值。 如果模型没有显示基准的改善，训练将停止。
restore_best_weights: 是否从具有监测数量的最佳值的时期恢复模型权重。 如果为 False，则使用在训练的最后一步获得的模型权重。

"""
"""
tensorflow.keras.callbacks.ModelCheckpoint 以某一频率保存Keras模型或模型权重
filepath	保存模型文件的路径。
monitor	要监测的指标
verbose	详细模式
save_best_only	保存最好的权重
mode	
save_weights_only	True：只保存权重
period	
"""
"""
https://vimsky.com/examples/usage/python-tf.keras.callbacks.TensorBoard-tf.html
tensorflow.keras.callbacks.TensorBoard
log_dir 保存要被 TensorBoard 解析的日志文件的目录路径。例如log_dir = os.path.join(working_dir, 'logs') 此目录不应被任何其他回调重用。
histogram_freq 计算模型层的激活和权重直方图的频率(以时期为单位)。如果设置为 0，则不会计算直方图。必须为直方图可视化指定验证数据(或拆分)。
write_graph 是否在 TensorBoard 中可视化图形。当 write_graph 设置为 True 时，日志文件可能会变得非常大。
write_images 是否编写模型权重以在 TensorBoard 中可视化为图像。
write_steps_per_second 是否将每秒的训练步数记录到 Tensorboard 中。这支持时代和批量频率记录。
update_freq 'batch' 或 'epoch' 或整数。使用 'batch' 时，在每批之后将损失和指标写入 TensorBoard。这同样适用于 'epoch' 。如果使用整数，比如说 1000 ，回调将每 1000 个批次将指标和损失写入 TensorBoard。请注意，过于频繁地写入 TensorBoard 会减慢您的训练速度。
profile_batch 分析批次以采样计算特征。 profile_batch 必须是非负整数或整数元组。一对正整数表示要分析的批次范围。默认情况下，分析是禁用的。
embeddings_freq 嵌入层将被可视化的频率(以时期为单位)。如果设置为 0，嵌入将不会被可视化。
embeddings_metadata 将嵌入层名称映射到文件的文件名的字典，在该文件中保存嵌入层的元数据。如果要对所有嵌入层使用相同的元数据文件，则可以传递单个文件名。
"""
callbacks_list = [
    tensorflow.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=10,
    ),
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath='./TFA-CLSTMNN/TFA-CLSTMNN_checkpoint_plus.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    tensorflow.keras.callbacks.TensorBoard(
        log_dir='./TFA-CLSTMNN/TFA-CLSTMNN_train_log'
    )
]
```

## Batch_size

`batch_size`（批大小）是指在**一次参数更新时**同时处理的样本数量。它在训练深度学习模型时扮演着重要的角色，具有以下几个重要的意义和原理：

1. **内存和计算效率**：较大的`batch_size`可以更好地利用计算资源，如GPU，并提高并行计算的效率。通过在每个参数更新步骤中处理更多的样本，可以减少数据加载和传输的开销，加快训练速度。

2. **梯度估计的稳定性**：在训练过程中，通过计算损失函数对模型参数的梯度来更新参数。较大的`batch_size`可以提供更稳定的梯度估计，因为它在计算梯度时考虑了更多的样本。这有助于减少梯度的方差，提高参数更新的稳定性。 

3. **参数更新的频率**：`batch_size`的选择决定了参数更新的频率。较小的`batch_size`意味着更频繁的参数更新，而较大的`batch_size`则意味着更少的参数更新。频繁的参数更新可以使模型更快地收敛，但也增加了计算开销。相反，较大的`batch_size`可以减少参数更新的频率，但可能导致更慢的收敛速度。

4. **正则化效果**：较小的`batch_size`**引入了一定程度的噪声**，因为每个小批次样本的统计特性可能会有所不同。这种噪声起到一种正则化的作用，有助于提高模型的泛化能力。较大的`batch_size`则可能降低了噪声的影响。

需要注意的是，较小的`batch_size`可能导致参数更新的方差较大，模型更容易陷入局部最小值。而较大的`batch_size`可能使模型更容易收敛到较差的局部最小值。

在选择适当的`batch_size`时，需要综合考虑计算资源、模型复杂度、数据集大小和模型性能等因素，并进行实验和调优来找到最佳的折中方案。

##  模型优化和压缩

当谈到深度神经网络的优化和压缩方法时，剪枝（Pruning）和知识蒸馏（Knowledge Distillation）是两种常用的技术。

1. 剪枝（Pruning）：剪枝是一种通过减少神经网络中不必要的连接或参数来减小模型大小和计算量的方法。这些不必要的连接或参数通常是在训练过程中学习到的，但对于模型的性能影响较小。剪枝可以分为结构剪枝和参数剪枝两种类型。

   - 结构剪枝：结构剪枝通过删除整个神经元、层或卷积核等结构来减小模型的规模。它通常基于某种剪枝策略，例如根据权重大小选择剪枝目标。
   - 参数剪枝：参数剪枝通过将模型中的参数设置为零或删除它们来减小模型的尺寸。参数剪枝通常依赖于设定一个阈值并对小于该阈值的参数进行剪枝。

   剪枝后的模型通常可以保持与原始模型相近的准确性，同时显著减少了模型的尺寸和计算需求。

2. 知识蒸馏（Knowledge Distillation）：知识蒸馏是一种将一个复杂模型（通常称为教师模型）的知识转移到另一个较简单模型（通常称为学生模型）的方法。这个过程通过训练学生模型以拟合教师模型的输出来实现。

   知识蒸馏利用了教师模型在训练数据上的预测分布，将其作为学生模型的目标分布。学生模型被训练以最小化与教师模型的目标分布之间的差异。通过这种方式，学生模型可以从教师模型中学习到更多的信息，包括模型之间的关系和数据集中的细节。

   知识蒸馏可以帮助减小模型的规模，并提高模型在测试集上的泛化能力。学生模型往往比教师模型更轻量级，适用于资源受限的环境或移动设备等场景。

这些方法都是用于优化和压缩深度神经网络的常见技术，旨在在保持模型性能的同时减少模型的尺寸和计算需求。

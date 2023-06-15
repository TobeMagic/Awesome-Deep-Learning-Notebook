## 模型建立

### 卷积神经网络

确定卷积层的最佳数量以及它们对其他参数的影响是一个挑战性的任务，通常需要进行实验和调整来找到最佳的模型架构。

一般而言，卷积层数量的选择可以基于以下因素进行评估：

1. 数据集大小和复杂程度：更大更复杂的数据集可能需要更深的卷积神经网络，以便提取更丰富的特征。较小的数据集则可能需要较浅的卷积神经网络，以避免过拟合。

2. 训练时长和计算资源：更深的卷积神经网络需要更长的训练时间和更多的计算资源。在限制时间和计算资源的情况下，可能需要权衡深度和精度。

3. 预训练模型的可用性：使用预训练模型可以减少训练时间并提高模型的精度。如果可用的预训练模型包含了与问题相关的卷积层，则可以考虑从这些层开始，然后通过微调来逐步优化模型。

除了卷积层的数量外，其他参数也会影响模型的性能。例如，卷积层的大小、步幅、填充等参数会影响特征图的大小和数量。池化层的类型、大小和步幅也会影响特征图的大小和数量，从而影响后续层的表现。因此，在设计卷积神经网络时，需要综合考虑这些参数，并进行实验和调整以找到最佳的模型结构。

> Q: CNN 1D与1至5个卷积模型测试性能获得的准确性分别为88.36%、89.48%、88.86%、87.96和86.89%。五个1D CNN层是最大的界限，因为这个层上的函数图的最小尺寸已经超过了。
>
> A: 这个问题涉及到卷积神经网络中的**感受野（receptive field）概念**。
>
> 在卷积神经网络中，每一层的卷积核实际上是对上一层特征图的局部区域进行处理，而**该局部区域的大小就是该层的感受野大小**。这意味着，随着层数的增加，感受野也会逐渐扩大。
>
> 在1D CNN中，每个卷积核只能查看其左右固定数目的元素，这个固定数目就是感受野。因此，通过堆叠多个1D CNN层，可以使得后面的层拥有更大的感受野，从而提取更全局的特征。
>
> 但是，当1D CNN层数过多时，每一层的输出的长度也会逐步缩小。这是因为，在1D CNN中，卷积操作将输入向量的每个元素映射到输出向量的一个元素，因此每次卷积操作都会减少向量长度。随着层数的增加，输出向量的长度也会逐渐缩小，最终可能会导致信息丢失，从而影响模型性能。
>
> 因此，作者在该问题中使用了1至5个1D CNN层进行测试，并发现5层是极限。作者指出，当使用5个1D CNN层时，**最后一层的输出长度已经非常短，无法再添加更多的卷积层**。因此，**作者不能通过增加层数来进一步提高模型性能，而必须尝试其他方法，如调整卷积核大小、池化方式等**，以达到更好的性能。

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


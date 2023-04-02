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
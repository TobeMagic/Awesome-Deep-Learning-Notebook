## Hyperparameter tuning(调优)

官方文档

https://keras.io/api/keras_tuner/

以下是KerasTuner的设计架构理念以及对应的特点和解决的问题的详细解释：

| 模块名称        | 主要特点和解决问题                                           |
| --------------- | ------------------------------------------------------------ |
| Tuners          | KerasTuner提供了几种不同的调参算法，称为"Tuners"。这些Tuners可以自动搜索给定的超参数空间，以找到最佳的模型配置。**不同的Tuners采用不同的搜索策略，例如随机搜索、网格搜索、贝叶斯优化等**。这些Tuners的主要特点是能够高效地在大型超参数空间中进行搜索，并逐步收敛到最佳配置。通过使用不同的Tuners，用户可以根据具体问题选择适合的搜索策略。 |
| HyperModel      | HyperModel是KerasTuner的核心概念之一。它定义了一个**可调整超参数的模型。用户可以通过继承HyperModel类，并实现其build方法，来定义自己的模型结构**。HyperModel的主要特点是能够根据不同的超参数配置动态地构建模型，从而实现模型的灵活性和可扩展性。 |
| HyperParameters | HyperParameters是KerasTuner用于定义和管理超参数的类。它允许用户**定义模型的超参数空间**，并指定每个超参数的可能取值范围。HyperParameters的主要特点是能够自动采样超参数配置，并将其传递给HyperModel进行模型构建和评估。通过使用HyperParameters，用户可以方便地定义和搜索超参数空间，从而优化模型性能。 |
| Trials          | Trials是KerasTuner用于跟踪每次实验的结果和配置的对象。每次Tuner搜索时，都会生成一个新的Trial对象，并记录该次搜索的超参数配置和相应的性能指标。Trials的主要特点是能够**保存和更新每次搜索的结果，并提供结果的可视化和分析工具**。通过使用Trials，用户可以更好地理解不同超参数配置的性能差异，并对搜索过程进行优化和调试。 |
| Oracle          | Oracle是KerasTuner中的一种策略，用于决定在每次搜索迭代中选择哪个超参数配置。Oracle的主要特点是能够基于之前的Trials结果，采用不同的策略来选择下一个要评估的超参数配置。例如，Oracle可以选择根据之前的结果进行贝叶斯优化，以加速搜索过程。通过使用Oracle，用户可以根据问题的特点和搜索需求，选择合适的超参数选择策略，提高搜索效率和模型性能。 |
| Callbacks       | Callbacks是KerasTuner中的回调函数机制，用于在每次搜索迭代中进行额外的操作和控制。例如，用户可以通过Callbacks在每次搜索迭代结束时保存模型权重、记录日志、调整学习率等。Callbacks的主要特点是能够在搜索过程中灵活地添加和调整功能，以满足用户的特定需求。通过使用Callbacks，用户可以更好地控制搜索过程，提高搜索效率和模型性能。 |

这些模块共同构成了KerasTuner的设计架构，并提供了以下特点和解决问题：

1. **高效的超参数搜索**：通过使用不同的Tuners，KerasTuner能够高效地在大型超参数空间中进行搜索，并逐步收敛到最佳配置。
2. **灵活的模型构建**：通过HyperModel的动态构建能力，KerasTuner实现了模型的灵活性和可扩展性，用户可以根据不同的超参数配置动态地构建模型，从而优化模型性能。
3. **方便的超参数定义和搜索**：通过HyperParameters类，KerasTuner提供了方便的接口来定义和搜索超参数空间，用户可以灵活地定义超参数的取值范围，以优化模型性能。
4. **结果跟踪和分析**：Trials对象可以跟踪每次搜索的结果和配置，提供结果的可视化和分析工具，帮助用户理解不同超参数配置的性能差异，并优化搜索过程。
5. **智能超参数选择**：通过Oracle的策略选择机制，KerasTuner可以根据之前的Trials结果，采用不同的策略选择下一个要评估的超参数配置，以提高搜索效率和模型性能。
6. **灵活的回调机制**：Callbacks机制允许用户在每次搜索迭代中进行额外的操作和控制，如保存模型权重、记录日志、调整学习率等，提供了灵活性和定制化的功能。

通过以上设计架构和特点，KerasTuner能够帮助用户高效地搜索超参数空间，优化模型性能，并提供灵活性和定制化的功能，以满足不同问题的需求。

非常抱歉之前的回答没有按照您的要求提供分开的代码示例。以下是每个模块的独立代码案例：

### 1. Tuners - 随机搜索示例
```python
from kerastuner.tuners import RandomSearch
from tensorflow import keras

# 定义模型构建函数
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 定义超参数搜索空间
hyperparameters = kerastuner.engine.hyperparameters.HyperParameters()
hyperparameters.Int('units', min_value=32, max_value=512, step=32)

# 定义随机搜索Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    hyperparameters=hyperparameters,
    directory='my_dir',  # 可选，保存搜索结果的目录
    project_name='my_project'  # 可选，搜索项目的名称
)

# 运行搜索
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# 获取搜索得到的最佳模型配置和超参数
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# 打印最佳模型配置和超参数
print("Best Model:")
print(best_model.summary())
print("Best Hyperparameters:")
print(best_hyperparameters.values)
```

### 2. HyperModel - 自定义模型结构示例
```python
from kerastuner import HyperModel
from tensorflow import keras

# 定义自定义HyperModel类
class MyHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

# 创建自定义HyperModel对象
hypermodel = MyHyperModel(num_classes=10)

# 定义随机搜索Tuner，并使用自定义HyperModel
tuner = RandomSearch(
    hypermodel.build,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='my_project'
)

# 运行搜索
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# 获取搜索得到的最佳模型配置和超参数
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# 打印最佳模型配置和超参数
print("Best Model:")
print(best_model.summary())
print("Best Hyperparameters:")
print(best_hyperparameters.values)
```

### 3. HyperParameters - 定义超参数空间示例
```python
from keras_tuner import HyperParameters
from tensorflow import keras

# 定义超参数空间
hyperparameters = HyperParameters()
hyperparameters.Int('units', min_value=32, max_value=512, step=32)

# 使用超参数空间构建模型
model = keras.Sequential()
model.add(keras.layers.Dense(units=hyperparameters.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 运行模型训练
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

### 4. Trials - 跟踪搜索结果和配置示例
```python
from kerastuner.tuners import RandomSearch
from tensorflow import keras

# 创建随机搜索Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='my_project'
)

# 运行搜索
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# 获取搜索得到的Trials对象，用于跟踪搜索结果和配置
trials = tuner.oracle.get_state()

# 打印每个Trials的超参数配置和评估结果
for trial in trials.trials:
    print("Trial ID:", trial.trial_id)
    print("Hyperparameters:", trial.hyperparameters.values)
    print("Metrics:", trial.metrics)
    print("---")
```

上述代码提供了每个模块的独立示例，您可以根据需要单独运行每个示例。请确保根据实际情况导入所需的库，并提供相应的训练数据（x_train、y_train）和验证数据（x_val、y_val）。







```python
from keras_tuner import HyperParameters as hp
from keras import layers
from keras.metrics import MSE
from keras.models import Model
from keras.losses import mse
# from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from keras.activations import leaky_relu
from kerastuner.tuners import RandomSearch


# 定义超参数空间
hp = hp()

# 构建模型的函数
def build_model(hp):
    inputs = layers.Input(shape=(12,1))
    x = layers.Conv1D(filters=hp.Int('units', min_value=8, max_value=32, step=8), kernel_size=hp.Int("kernel", min_value=2, max_value=4, step=1))(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(hp.Int('units', min_value=8, max_value=64, step=16))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(hp.Int('units', min_value=8, max_value=32, step=8), activation=leaky_relu)(x)
    output = layers.Dense(1, activation="relu")(x)

    model = Model(inputs, output)
    model.compile(optimizer='rmsprop', loss=mse)
    return model

# 创建随机搜索Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    hyperparameters=hp,
    directory='cnn-lstn',
    project_name='my_project',
    seed = 42
)

# 运行搜索
tuner.search(train_data, train_target, epochs=70, validation_data=(val_data, val_target), shuffle=False)

# 获取搜索得到的最佳超参数配置
best_hyperparameters = tuner.get_best_hyperparameters()[0]
best_units = best_hyperparameters.get('units')
best_kernel = best_hyperparameters.get('kernel')

print("Best Hyperparameters:")
print("units:", best_units)
print("kernel:", best_kernel)

# 获取最佳模型
best_model = tuner.get_best_models(num_models=1)[0]
# 使用最佳超参数配置来重新编译模型
best_model.compile(optimizer='rmsprop', loss=mse)
# 在完整的训练集上重新训练模型
best_model.fit(train_data, train_target, epochs=70, shuffle=False)
model.summary()
```


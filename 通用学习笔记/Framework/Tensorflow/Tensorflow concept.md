### 模型动态图图最佳实践 (Eager)

####　**`Static Graph` ＆`Eager Execution&mode` **

在TensorFlow 1中，静态图（Static Graph）是一种表示**计算流程**的概念。它使用**数据流图（Data Flow Graph）**来描述计算任务，其中节点表示操作（operations），边表示数据流动。

具体而言，在TensorFlow 1中，你需要**先定义一个计算图，并将所有操作和变量添加到该图中**。然后，通过**运行会话（Session）来执行这个静态图。**这种方式使得 TensorFlow 可以**对整个计算过程进行优化和编译**，并允许并行执行多个操作。（C++引擎）

相比之下，在TensorFlow 2中引入了 Eager Execution （即即时执行）。Eager Execution 是一种命令式编程环境，可以立即评估操作并返回结果。与静态图不同的是，在 Eager Execution 模式下，**每条语句都会被立即执行，并且可以实时查看结果**。（能够很好的结合Python本身，但在面对python运行时下性能能收到影响（自动并行化、GPU 加速），后面说的`@tf.function`便是解决改性能问题的，此外**启用 Eager Execution 后可能会增加某些模型的开销**。)

Eager Execution 的好处包括**更直观的代码编写、更容易调试、更灵活地处理控制流**（*自然的控制流* - 使用 Python 而非计算图控制流）等。它消除了在 TensorFlow 1 中构建和运行静态图所需的繁琐步骤，并使得机器学习开发变得更加交互式和直观。（怎么有点像是解释型语言和编译型语言的区别？）

> 要注意的是，在 TensorFlow 2 中，**默认启用 Eager Execution** 。但如果需要使用类似于 TensorFlow 1 的静态图模式，则也可以手动将其启用。
>
> ```python
> import tensorflow as tf
> 
> # 禁用 Eager Execution
> tf.compat.v1.disable_eager_execution()
> ```
>
> 请注意，这种模式下，你需要使用 `tf.compat.v1.Session()` 来创建会话，并使用 `tf.compat.v1.placeholder()` 来定义占位符等 TensorFlow 1 的操作。
>
> 启用静态图模式可能对某些特定的高级功能和新特性不兼容，因为它们是在 Eager Execution 模式下引入的。因此，在使用静态图模式之前，请确保你了解其可能的限制和影响，并确保它满足你的需求。

综上所述：

- 在 TensorFlow 1 中，数据流图是指用于描述计算任务的静态图，需要通过会话进行执行。
- 在 TensorFlow 2 中，默认启用 Eager Execution ，它允许立即执行操作并返回结果，使得代码编写更直观和灵活。

#### **`@tf.function` 装饰器**

`@tf.function` 是 TensorFlow 中的一个装饰器（decorator），它的作用是**将 Python 函数转换为 TensorFlow 的计算图**，并且可以提供更高效的执行。

以下是 `@tf.function` 装饰器的主要目的和作用：

1. **提升性能**：通过将函数转换为 TensorFlow 计算图，可以**利用 TensorFlow 的自动并行化、GPU 加速**等功能，从而实现更高效地执行。这**对于大规模数据集或复杂模型特别有益**。

2. **实现符号式编程**：TensorFlow 使用静态计算图进行工作。使用 `@tf.function` 将函数转换为计算图后，该计算图中包含了所有操作和控制流程，使得代码更具可视化和理解性。这种方式被称为符号式编程，与命令式编程相比，在一些情况下可能会带来性能上的优势。

3. **支持跨平台部署**：由于将函数转换为计算图后，**不再依赖原始 Python 代码本身运行时环境**，因此可以方便地在其他平台上部署、导出或共享训练好的模型。

4. **自动微分**：当使用 `@tf.function` 转换函数时，**默认情况下会启用自动微分功能**。这意味着你可以轻松地**获取关于输入变量的梯度信息**（微分方程），用于训练过程中的优化。

需要注意以下几点：

- `@tf.function` 装饰器**只能应用于满足特定条件的函数**。它要求函数**内部使用 TensorFlow** 的操作（例如**张量运算、控制流等**），并且**不包含 Python 语法或控制流结构（如循环、条件语句）**。这是因为计算图在转换时需要**静态确定**所有操作，并且无法动态执行 Python 代码。

- 如果你想了解哪些函数可以被装饰，请参考 TensorFlow 官方文档中关于 `@tf.function` 的说明和示例。

官方文档：https://www.tensorflow.org/guide/eager?hl=zh-cn#computing_gradients

### tensorflow 分类默认阈值

> 深度学习网络中的结果是 0 1 ，但预测出来的是概率，那评估加过accuary是怎么出来的？ 怎么确定大于多少概率为1 或0？

在深度学习中，常见的二分类任务中，模型的输出是一个表示概率的连续值，通常在0到1之间。但是，我们通常需要将这些概率转化为二进制的0或1以进行评估和判断。

在 TensorFlow 中，**模型的默认阈值通常是**0.5，即当模型输出的概率大于或等于0.5时，将其判定为1；当概率小于0.5时，将其判定为0。

具体到代码中，模型的输出将是一个概率向量或矩阵，其中每个元素表示样本属于不同类别的概率。你可以使用以下代码来查看模型的输出，并根据阈值进行二分类预测：

```python
import tensorflow as tf

# 假设模型的输出是一个概率向量或矩阵
probabilities = model.predict(x_test)

# 将概率转化为二进制的0或1，根据默认阈值0.5
predictions = (probabilities >= 0.5).astype(int)
```

在上述代码中，`model` 是你的 TensorFlow 模型，`x_test` 是用于测试的输入数据。`probabilities` 是模型的输出概率向量或矩阵，`predictions` 是根据阈值0.5进行的二分类预测结果。

需要注意的是，虽然0.5 是常见的默认阈值，但在特定的情况下，根据问题的需求，你可能需要调整阈值来优化模型的性能。这可以根据实际情况在代码中进行调整。


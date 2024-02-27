官方文档：https://fairseq.readthedocs.io/en/latest/overview.html
Github: https://github.com/facebookresearch/fairseq/tree/main

Numba 是一个适用于 Python 代码的开源式即时编译器。借助该编译器，开发者可以使用标准 Python 函数在 CPU 和 GPU 上加速数值函数。

为了提高执行速度，Numba 会在执行前立即将 Python 字节代码转换为机器代码。 

Numba 可用于使用可调用的 Python 对象（称为修饰器）来优化 CPU 和 GPU 功能。修饰器是一个函数，它将另一个函数作为输入，进行修改，并将修改后的函数返回给用户。这种模组化可减少编程时间，并提高 Python 的可扩展性。

Numba 还可与[ NumPy](https://numpy.org/) 结合使用，后者是一个复杂数学运算的开源 Python 库，专为处理统计数据而设计。调用修饰器时，Numa 将 Python 和/或 NumPy 代码的子集转换为针对环境自动优化的字节码。它使用 [LLVM](http://llvm.org/)，这是一个面向 API 的开源库，用于以编程方式创建机器原生代码。Numba 针对各种 CPU 和 GPU 配置，提供了多种快速并行化 Python 代码的选项，有时仅需一条命令即可。与 NumPy 结合使用时，Numba 会为不同的数组数据类型和布局生成专用代码，进而优化性能。

参考文章：
https://www.nvidia.cn/glossary/data-science/numba/
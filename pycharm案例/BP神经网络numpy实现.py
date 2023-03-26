# Python中使用numpy库实现一个简单的bp神经网络的示例代码。该网络包含一个输入层、一个隐藏层和一个输出层。
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重矩阵
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # sigmoid激活函数的导数
        return x * (1 - x)

    def forward(self, input):
        # 前向传播
        self.hidden = self.sigmoid(np.dot(input, self.weights1))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backward(self, input, output, target, learning_rate):
        # 反向传播
        error = target - output
        output_delta = error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        self.weights2 += self.hidden.T.dot(output_delta) * learning_rate
        self.weights1 += input.T.dot(hidden_delta) * learning_rate

    def train(self, input, target, learning_rate, epochs):
        # 训练神经网络
        for epoch in range(epochs):
            output = self.forward(input)
            self.backward(input, output, target, learning_rate)

    def predict(self, input):
        # 使用训练后的网络进行预测
        return self.forward(input)


# 创建神经网络实例
nn = NeuralNetwork(2, 3, 1)

# 准备训练数据
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

# 训练神经网络 XOR 异或
nn.train(input, target, learning_rate=0.1, epochs=10000)

# 进行预测
print(nn.predict(np.array([0, 1])))  # 输出应该接近1
print(nn.predict(np.array([1, 1])))  # 输出应该接近0

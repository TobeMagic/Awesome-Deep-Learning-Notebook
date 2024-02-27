官方文档：https://fairseq.readthedocs.io/en/latest/overview.html
Github: https://github.com/facebookresearch/fairseq/tree/main

Fairseq是由Facebook AI Research开发的一个**序列到序列模型工具包**（它是对pytorch的上层封装，其基础代码也是通过pytorch编写）。它支持各种模型架构，包括卷积神经网络（CNNs）、循环神经网络（RNNs）和Transformer模型。

Fairseq已经被广泛应用于自然语言处理和语音识别领域，并提供各种序列建模论文的参考实现。特性如下：

<img src="https://markdown-1311598839.cos.ap-nanjing.myqcloud.com/img/image-20240127111020996.png" alt="image-20240127111020996" style="zoom: 50%;" />

并且还提供用于翻译和语言建模的预训练模型以及便捷的界面（非常方便）：

```python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

See the PyTorch Hub tutorials for [translation](https://pytorch.org/hub/pytorch_fairseq_translation/) and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.
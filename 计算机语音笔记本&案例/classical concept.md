## 最佳实践

### 计算频率范围

使用快速傅里叶变换（FFT）时，可以通过以下步骤来查看频率：

1. 对信号进行离散傅里叶变换（DFT），得到信号的频域表示。

2. 计算每个频率分量的振幅和相位。

3. 将振幅转换为单位为 dB 的对数尺度，以便更容易地观察不同频率成分之间的差异。

4. 绘制频谱图，其中 x 轴表示频率，y 轴表示振幅或功率。

如果需要确定某个频率的具体数值，可以将该频率的索引与采样率相乘，即可得到该频率在Hz中的值。例如，假设采样率为1000Hz，而某个频率在DFT结果中的索引为10，则该频率的实际值为10*1000/1024=9.77 Hz。 

`np.fft.fftfreq()`函数是`NumPy`库中用于计算离散傅里叶变换（DFT）的快速傅里叶变换（FFT）结果对应的频率的函数之一。该函数的语法如下：

```python
numpy.fft.fftfreq(n, d=1.0)
```

其中，参数`n`表示信号长度，参数`d`表示采样周期，单位为秒，默认值为1.0秒。

该函数会返回一个长度为`n`的`NumPy`数组，包含由FFT结果对应的频率。具体来说，该函数会生成以0为中心的实数数组，其范围从负Nyquist频率到正Nyquist频率，即[-1/(2*d), 1/(2*d)]。因此，可以通过将该数组乘以采样频率来得到以Hz为单位的频率值。

需要注意的是，该函数返回的频率值是针对实数FFT结果的。如果使用复数FFT，则需要自行计算对应的频率值。

以下是一个简单的示例。假设我们有一个长度为8的信号，采样周期为0.1秒，其代码如下：

``` python
import numpy as np

sig_length = 8
sample_period = 0.1

freqs = np.fft.fftfreq(sig_length, d=sample_period)

print(freqs)
```

输出结果如下：

```
[ 0.     0.125  0.25   0.375 -0.5   -0.375 -0.25  -0.125]
```

可以看出，该函数返回了一个长度为8的数组，其中心点是0，表示直流信号。其他频率值则从负Nyquist频率到正Nyquist频率递增。如果我们将其乘以采样频率（10Hz），则可以得到相应的频率值：

``` python
freqs_hz = freqs * 10

print(freqs_hz)
```

输出结果如下：

```
[ 0.     1.25   2.5    3.75  -5.    -3.75  -2.5   -1.25 ]
```

因此，可见计算FFT结果对应的频率时，`np.fft.fftfreq()`函数非常实用。

> 傅里叶变换可以将一个信号从时域转换到频域，即将信号表示为不同频率的正弦和余弦波的加权和。在计算机中，傅里叶变换可以通过快速傅里叶变换（FFT）来实现。
>
> 在使用 FFT 进行频率域分析时，我们需要知道**每个频率对应着什么样的幅值**，因此就需要通过 `np.fft.fftfreq` 函数计算出频率信息。
>
> 具体来说，`np.fft.fftfreq` 函数用于生成一组表示频率的数字，这些数字对应于 `np.fft.fft` 返回的复数数组的各个元素。例如，如果 `np.fft.fft` 返回一个长度为 N 的复数数组，则 `np.fft.fftfreq(N)` 会返回一个长度为 N 的一维数组，该数组中的每个元素都是一个浮点数，**表示对应于该位置的复数值的频率**。
>
> 需要注意的是，`np.fft.fftfreq` 函数返回的频率值默认以“单位周期数”为单位，即每个周期含有一个完整的信号。如果需要以其他单位表示频率，可以通过除以采样率来进行换算。

**数学原理**

np.fft.fftfreq用于计算信号的频率。在语音信号处理中，可以使用离散傅里叶变换（DFT）将时域信号转换为频域信号，并对其进行进一步的分析和处理，如滤波、降噪等。

假设我们有一个长度为N的时域信号x(n)，其中n表示时间序列。使用DFT，我们可以将信号转换为由复数表示的频域信号X(k)，其中k表示频率序列。DFT的公式如下：

$$ X(k)=\sum_{n=0}^{N-1} x(n) e^{-2\pi i kn/N} $$

其中i表示虚数单位，k取值范围为0到N-1，n取值范围也为0到N-1。X(k)表示信号在第k个频率点的幅度和相位信息。

np.fft.fftfreq函数用于计算每个频率点对应的实际频率值。它的公式如下：

$$ f_k=k\times \frac{f_s}{N} $$

其中fk为第k个频率点对应的实际频率值，fs为信号的采样频率，N为信号的采样点数。

因此，`np.fft.fftfreq`返回的是一个长度为N的numpy数组，包含从0到N-1的所有非负整数，乘以信号的采样频率并除以N，得到对应的频率值。它可以帮助我们将DFT输出的频率轴转化为实际的频率值。

总之，`np.fft.fftfreq`函数结合离散傅里叶变换可以在语音信号处理中帮助我们分析信号的频域特征。

> 因为在数字信号处理中，离散时间信号的频域表示通常使用离散傅里叶变换（Discrete Fourier Transform，DFT）来实现。在进行DFT时，我们需要将离散时间信号转换为离散频率信号，也就是将信号从时域表示转换到频域表示。在离散频率信号中，每个元素对应一个特定的频率，而这个频率值与该元素在数组中的索引有关。具体而言，它等于该元素的索引乘以采样频率除以DFT长度，即：
>
> 频率 = 索引 x 采样频率 / DFT长度
>
> 因此，如果我们知道了离散频率信号元素的索引和采样率，就可以计算出相应的频率值。
>
> 这个公式是因为在进行DFT时，我们将时间域的N个采样点转换为频率域中的N个离散频率。这些离散频率的数量等于采样点数N。这些离散频率的范围从0到采样率（sampling rate）之间，其中采样率等于采集信号时每秒钟采样的点数。
>
> 因此，**我们通常将频率划分为N个相等的部分，每个部分表示一个离散频率。这些离散频率的值可以通过对索引进行归一化来计算，即将索引乘以采样频率除以DFT长度。这就是为什么频率等于索引 x 采样频率 / DFT长度的原因。**

### 幅度转分贝

`librosa.amplitude_to_db` 函数的作用是将幅度谱转换为分贝谱。这个函数实际上包含了两步转换：

1. 将输入的幅度谱取对数（以10为底），得到一个对数幅度谱。
2. 将对数幅度谱乘以一个系数20，再加上一个参考值（默认是1e-5）。

因此，可以看出 `librosa.amplitude_to_db` 的原理确实是将幅度谱取对数（以10为底）。而这样做的原因是因为人耳对音频信号的感知是呈现出对数关系的，因此将幅度谱转换为对数幅度谱的方式更符合人类听觉的特性。

至于 `ref` 参数，它控制了输出分贝谱中的参考值，默认为 $1e-5$。当 `ref` 取较小的值时，输出的分贝谱变化范围会扩大，反之变化范围会缩小。`red` 参数则控制是否对分贝谱进行归一化处理，即将输出的分贝谱限制在某个特定的范围内。如果 `red=True`，那么输出的分贝谱值会被压缩到 $[0, 1]$ 的范围内。

### AudioSegment  类

AudioSegment是Python中的一个类，它是由PyDub库提供的。 它用于处理音频文件，并提供了多种音频操作方法。

对于一段音频文件，可以使用AudioSegment类进行读取并存储为对象。 一旦音频被加载到对象中，便可以对其进行多种操作，如裁剪、混音、调整速度和音量等。

以下是AudioSegment类的一些常见操作：

1. 读取音频文件：

```python
from pydub import AudioSegment

sound = AudioSegment.from_file("audio.mp3", format="mp3")
```

这段代码将从名为“audio.mp3”的MP3文件中读取音频数据并将其存储在“sound”对象中。

2. 保存音频文件：

```python
sound.export("output.wav", format="wav")
```

这段代码将把“sound”对象中的音频数据保存为名为“output.wav”的WAV文件。

3. 裁剪音频：

```python
new_sound = sound[start_time:end_time]
```

这段代码将从“sound”对象中提取出从“start_time”到“end_time”时间段内的音频数据，并将其存储在“new_sound”对象中。

4. 混音音频：

```python
mixed_sound = sound1.overlay(sound2)
```

这段代码将混合在“sound1”和“sound2”对象中包含的音频数据，并将结果存储在“mixed_sound”对象中。

5. 调整音频速度：

```python
speed_adjusted_sound = sound.speedup(playback_speed)
```

这段代码将更改“sound”对象的播放速度，使其倍增或减少。 结果存储在“speed_adjusted_sound”对象中。

6. 调整音频音量：

```python
louder_sound = sound + 10
```

这段代码将调整“sound”对象的音量，并将其增加10分贝。 结果存储在“louder_sound”对象中。

以上是AudioSegment类的一些基本操作。 PyDub库提供了许多其他有用的方法和工具，可以在处理音频文件时使用。

要获取`AudioSegment`对象的音频数据，可以使用`get_array_of_samples()`方法，该方法返回一个NumPy数组，其中包含该音频文件中所有样本的值。您可以像这样使用它：

```python
from pydub import AudioSegment

# 加载音频文件
audio = AudioSegment.from_file("example.wav", format="wav")

# 获取对应的音频数据
samples = audio.get_array_of_samples()

# 您现在可以使用许多NumPy函数来处理和分析音频数据
```

请注意，此方法返回的数组类型取决于Python版本和PyDub库的版本。在较新版本的PyDub（例如v0.25.1）与Python 3中，返回的类型应为`numpy.ndarray`。在旧版本的PyDub或Python 2中，则可能返回`array.array`类型。

### 帧数转时间

librosa.frames_to_time函数用于将时间轴上的帧索引转换为对应的时间值。其原理是根据音频信号的采样率和帧跨度，计算每个帧的持续时间，并累加得到该帧结束的时间点。

具体地，假设音频信号的采样率为sr，帧跨度为hop_length，则每个帧的持续时间为hop_length / sr，那么第n帧的起始时间为n * hop_length / sr，结束时间为(n+1) * hop_length / sr。因此，frames_to_time函数就是根据这些计算公式将帧索引映射到对应的时间值上的。

### 音频归一化最佳实践（特征提取前后归一化)

通常情况下，对原始波形数据进行归一化是在特征提取之前进行的。

首先，原始波形数据通常具有不同的振幅范围，归一化可以将振幅范围限制在一定的范围内，以便后续的处理和分析。常见的归一化方法是将波形数据线性缩放到一个特定的范围，例如将振幅值映射到-1至1之间。

然后，在归一化后，可以对波形数据进行特征提取。特征提取是从波形数据中提取出具有代表性的特征，用于后续的分析和处理，比如语音识别或音频分类等任务。常见的特征提取方法包括短时傅里叶变换（Short-Time Fourier Transform, STFT）、梅尔频率倒谱系数（Mel-Frequency Cepstral Coefficients, MFCC）等。

最后，对提取的特征进行归一化可以进一步处理特征数据的范围和分布。这种归一化通常是针对特征维度进行的，而不是针对整个波形数据。常见的特征归一化方法包括零均值归一化（Zero-mean normalization）和单位方差归一化（Unit variance normalization）等。

总结起来，一般的流程是：原始波形数据归一化 -> 特征提取 -> 特征归一化。这样可以确保在音频处理过程中数据的范围和分布得到适当的控制和调整。

## 核心概念

### 相关语音处理库

以下是一些经典常用的用于语音处理的库：

1. `librosa`：用于分析和处理音频信号的Python库，支持许多常见的音频格式。
2. `Pydub`：一个简单易用的Python库，用于处理音频文件，可以读取和写入多种格式的音频文件。
3. `SpeechRecognition`：一个支持多种语音识别引擎的Python库，可以将音频转换为文本。
4. `Soundfile`：用于读取和写入许多常见音频格式的Python库，具有简单易用的API。
5. ``NumPy``：用于数值计算和科学计算的Python库，可以用于处理音频信号数据。
6. `SciPy：Python`中的另一个科学计算库，提供了许多用于信号处理的函数。
7. `TensorFlow：Google`开发的深度学习框架，支持语音识别、音频处理等应用。
8. `Kaldi`：一个流行的开源语音识别工具包，用于训练和测试语音识别模型。
9. `HTK：Hidden `Markov Model Toolkit，另一个流行的语音识别工具包，用于语音信号的特征提取、建模和识别。
10. `Praat`：一个用于语音分析的软件，可以进行语音信号的基本处理和分析，例如提取基频、共振峰等。

### 音频纹理

纹理的定义本质上是相当不精确的，但主要指向音频信号中**存在的刺耳/粗糙度**。如果声音具有统一的长期属性和较长的注意力范围，那么它就可以成为纹理。根据这个定义，我们可以将诸如单次拍手之类的声音标记为非纹理音频，将掌声标记为纹理音频。

> 如果音频声音是长期的、非正弦的、随机的，但本质上仍然是均匀的，在整个信号中保持均匀的能量，具有较差或没有谐波内容，表现出类似噪声的特征，没有特定的起始点，并且有时可以引发人类的情绪，例如恐惧、兴奋、快乐等。

文献研究表明，在某些病理条件下，音频信号固有的纹理会发生变化。例如 COVID-19 声音、特定语言障碍 (SLI) 语音、成人病理性语音和婴儿哭声。

1. 选择 COVID-19 咳嗽/呼吸音作为案例研究，因为这些声音表现出与其他类型的呼吸咳嗽/呼吸音不同的固有纹理 。研究发现，COVID-19 阳性者气道干燥，导致咳嗽/呼吸音有些嘶哑。这种声音嘶哑可以通过音频纹理来表示。
2. 从健康的言语中筛选 SLI 言语。与利用声学特征或深度网络进行分类的现有方法不同，我们提出提取病理语音中存在的音频纹理[13]。
3. 根据婴儿哭声分为5类：饥饿、胃痛、疲倦、不适和打嗝。婴儿哭声满足我们在第一节中定义的音频纹理的几个属性。婴儿哭声表现出类**噪声特征、伪装事件和声音同质**。因此，音频纹理可用于模拟婴儿的哭声。
4. 成人病理性言语与健康言语进行分类。由于某种程度的病态，语音的纹理变得粗糙，并且与健康语音样本中存在的纹理显着不同。

### 音频波纹图为什么有负值

音频波纹图用于表示声音信号的振幅随时间的变化情况。在一些音频波纹图中，振幅值可能会出现负值。

这是因为音频信号是由正负压力变化构成的。声音传播是通过空气分子的振动传递的，当空气分子受到压力增加时，它们会向外扩散，形成正压力区域；而当空气分子受到压力减小时，它们会向内收缩，形成负压力区域。

在音频波纹图中，正值表示正压力区域，负值表示负压力区域。这些正负值反映了声音信号的振幅变化情况。当音频波纹图中出现负值时，意味着声音信号在某些时间点上的振幅是负的，即声音信号的振动方向与参考点的方向相反。

需要注意的是，音频波纹图中的负值并不代表声音本身是负的或消极的，它只是表示声音信号在某些时间点上的振幅方向与参考点相反。

### 语音识别系统标准帧长

帧长必须足够长，使得一帧信号内有足够多的周期；但又不能太长，因为需要保证**一帧内的信号基本平稳**。人声的基频范围下限在100 Hz左右（有些男声会更低），换算成周期是10 ms，所以一般帧长都选在20 ~ 50 ms。在这个范围内选一个比较“整”的数，比如20、25、40、50，都很常见。

常见的采样率系列有8 kHz、16 kHz和11025 Hz、22050 Hz、44100 Hz。

参考：
https://www.zhihu.com/question/50044248/answer/119118646

### 其他

人类语音的频率范围小于 8 kHz，采样率大于16kHz即可


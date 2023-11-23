# 特征提取

当涉及到音频信号处理中的四种特征提取方法：GFCC（Gammatone Frequency Cepstral Coefficients）、IMFCC（Improved Multi-Frequency Cepstral Coefficients）、EMFCC（Enhanced Mel Frequency Cepstral Coefficients）和MFCC（Mel Frequency Cepstral Coefficients），让我们逐个进行详细解释，并提供相应的原理和模板代码。使用库（Spectrogram Analysis FrameWork) `spafe`  (更具业界可以知道最常用的是MFCC，我们只需要用改进的和用Gmamatone的尝试即可)

## MFCC（Mel Frequency Cepstral Coefficients）：

- 原理：MFCC 是一种常见的音频特征提取方法，它模拟人类听觉系统对声音的感知。主要包含以下步骤：

   1. 预加重：通过应用一个高通滤波器来平衡语音信号的频谱。
   2. 分帧：将音频信号分割成小的时间窗口。
   3. 加窗：对每个时间窗口应用窗函数（如汉明窗）以减少频谱泄露。
   4. 快速傅里叶变换（FFT）：计算每个窗口的频谱。
   5. 梅尔滤波器组：应用一组三角滤波器，这些滤波器在梅尔频率尺度上均匀分布，以模拟人耳的感知特性。
   6. 对数压缩：将频谱转换为对数刻度，以增加特征的区分度。
   7. 离散余弦变换（DCT）：应用DCT将对数频谱转换为倒谱系数，保留较少的高频信息。

- 模板代码：

   ```python
   import librosa
   
   # 加载音频文件
   audio_file = 'path/to/audio.wav'
   y, sr = librosa.load(audio_file)
   
   # 计算MFCC特征
   mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   
   # 打印MFCC特征
   print('MFCC:')
   print(mfcc)
   ```

## IMFCC（Improved Multi-Frequency Cepstral Coefficients）：

- IMFCC在MFCC的基础上进行了一些改进，以提高其表达能力和鲁棒性。

   主要的改进包括以下几点：

   1. 动态参数：IMFCC引入了动态参数，例如速度和加速度，来描述信号的时序变化。这些动态参数能够捕捉到音频信号的时间演化信息，从而提供更多关于声音变化的上下文信息。
   2. 预加重滤波器：预加重滤波器被用于增强高频信号，以提高特征提取的准确性。IMFCC采用了改进的预加重滤波器设计，使得音频信号中的高频成分更加突出，有助于后续的特征提取过程。
   3. 改进的倒谱提取：IMFCC使用改进的倒谱提取方法，以减小噪声对特征提取的影响。这种改进可以提高MFCC的鲁棒性，使其对于环境噪声和信号失真具有更好的适应性。

- 模板代码：

   ```python
   import numpy as np
   import scipy.fftpack as fftpack
   import librosa
   
   # 加载音频文件
   audio_file = 'path/to/audio.wav'
   y, sr = librosa.load(audio_file)
   
   # 预加重
   pre_emphasis = 0.97
   emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
   
   # 分帧
   frame_length = int(0.025 * sr)  # 25ms
   frame_step = int(0.01 * sr)  # 10ms
   frames = librosa.util.frame(emphasized_signal, frame_length=frame_length, hop_length=frame_step)
   
   # 加窗
   window = np.hamming(frame_length)
   windowed_frames = frames * window[:, np.newaxis]
    # 计算功率谱
   power_spectrum = np.abs(np.fft.rfft(windowed_frames, axis=1)) ** 2
    
   # 计算IMFCC特征
   imfcc = fftpack.dct(10 * np.log10(power_spectrum), axis=1, type=2, norm='ortho')
   
   # 打印IMFCC特征
   print('IMFCC:')
   print(imfcc)
   ```

## EMFCC（Enhanced Mel Frequency Cepstral Coefficients）：

- 原理：EMFCC是对MFCC的增强版本，它在MFCC的基础上引入了额外的增强参数，主要包括以下步骤：

   1. 预加重：通过应用一个高通滤波器来平衡语音信号的频谱。
   2. 分帧和加窗：将音频信号分割成时间窗口并应用窗函数。
   3. 幅度谱计算：通过应用快速傅里叶变换（FFT）计算每个窗口的幅度谱。
   4. 梅尔滤波器组：应用一组三角滤波器，这些滤波器在梅尔频率尺度上均匀分布，以模拟人耳的感知特性。
   5. 对数压缩：将频谱转换为对数刻度，以增加特征的区分度。
   6. 动态特征（Delta和Delta-Delta）**：通过计算当前帧与相邻帧之间的差异来捕获频谱随时间的变化**。
   7. 合并特征：**将MFCC特征与`动态特征`连接起来形成最终的EMFCC特征**。

EMFCC的改进主要包括以下方面：

1. 增加高阶倒谱系数：除了常规的MFCC系数，EMFCC还引入了高阶倒谱系数。这些高阶倒谱系数能够提供更多频率分辨率和更丰富的音频特征信息。
2. 非线性变换：EMFCC采用非线性变换方法对MFCC系数进行处理。这种非线性变换可以增强特征之间的差异性，使得提取的特征更具判别性。
3. 去相关技术：EMFCC使用去相关技术来减小特征之间的相关性。通过消除特征之间的冗余信息，可以提高特征的独立性和表达能力。

- 模板代码：

   ```python
   import librosa
   
   # 加载音频文件
   audio_file = 'path/to/audio.wav'
   y, sr = librosa.load(audio_file)
   
   # 计算MFCC特征
   mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   
   # 计算动态特征（Delta和Delta-Delta）即相邻差分,得到动态变化
   delta_mfcc = librosa.feature.delta(mfcc)
   delta2_mfcc = librosa.feature.delta(mfcc, order=2)
   
   # 合并MFCC和动态特征
   emfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc]) # 一阶和二阶动态特征垂直堆叠在一起，形成一个新的特征矩阵emfcc。	
   
   # 打印EMFCC特征
   print('EMFCC:')
   print(emfcc)
   ```

> `librosa.feature.delta()`函数用于计算给定特征矩阵的**一阶或二阶差分**（Delta和Delta-Delta）。
>
> Delta是一种描述特征随时间变化的衍生特征。它通过计算相邻帧之间的差异来捕捉音频特征的动态信息，例如声音的速度变化。Delta可以帮助提取语音、音乐等音频信号中的快速变化特征。
>
> Delta的计算过程如下：
>
> 1. 假设有一个大小为`(n_features, n_frames)`的特征矩阵X，其中`n_features`是特征的数量，`n_frames`是帧的数量。
> 2. 对于每个特征维度i，在每个时间帧t上计算差分值： `delta_X[i, t] = X[i, t+1] - X[i, t-1]` 要注意边界情况，当`t`为第一个或最后一个帧时，需要使用适当的边界处理方式。
> 3. 得到一个新的特征矩阵`delta_X`，其大小为`(n_features, n_frames)`。
>
> 与Delta类似，Delta-Delta是对Delta特征进行进一步差分计算得到的二阶差分特征。Delta-Delta的计算过程与Delta相似，只是在Delta的基础上再次应用差分运算。
>
> `librosa.feature.delta()`函数的调用方式如下：
>
> ```python
> delta_X = librosa.feature.delta(X, width=N, order=1, axis=-1, mode='interp', **kwargs)
> ```
>
> 参数说明：
>
> - `X`：输入特征矩阵（例如MFCC特征矩阵）。
> - `width`：差分计算窗口的宽度。默认值为 9，表示使用前后各4个帧进行差分计算。
> - `order`：差分的阶数。`order=1`表示一阶差分（Delta），`order=2`表示二阶差分（Delta-Delta）。
> - `axis`：指定在哪个轴上进行差分计算。默认值为-1，表示最后一个轴。
> - `mode`：指定边界处理方式。默认为'interp'，表示使用插值填充边界值。
>
> `librosa.feature.delta()`函数返回计算得到的差分特征矩阵。
>
> 总结起来，Delta特征是通过计算音频特征在时间上的差异来捕捉音频动态变化的衍生特征。它可以帮助提取音频信号中的快速变化特征。Delta-Delta则是对Delta特征再次进行差分计算得到的二阶差分特征。


## GFCC（Gammatone Frequency Cepstral Coefficients）：

- 原理：GFCC是一种**基于Gammatone滤波器的特征提取方法**，它模拟人耳的听觉特性对声音的感知，它结合了组延迟和倒谱分析的概念，可以有效地捕捉音频信号中的时变频谱特征。主要包含以下步骤：

   1. 分帧：首先，将音频信号分成短时间片段，通常每个片段为20-40毫秒，这样可以保证信号在每个时间段内是基本稳定的。
   2. 傅里叶变换：对每个时间片段进行傅里叶变换，将时域信号转换为频域信号。这样可以得到每个时间片段内的频谱信息。
   3. 组延迟计算：对于每个频谱帧，计算其组延迟。组延迟表示信号在频域上的相位变化速度。计算组延迟可以通过计算相邻帧之间的相位差来实现。
   4. 倒谱分析：对于每个频谱帧，应用倒谱分析。倒谱是指对频谱取对数然后进行傅里叶逆变换。**倒谱分析可以减小谱峰之间的差异，使得特征更加稳定**。
   5. 梅尔滤波器组：将倒谱系数输入到梅尔滤波器组中，梅尔滤波器组通常包含一系列三角形滤波器，用于对频谱进行滤波。每个滤波器对应一个特定的频率范围，可以提取出相应频率范围内的能量。
   6. 能量整合：对于每个滤波器输出的能量进行积分或平均，得到每个滤波器输出的总能量。
   7. 倒谱提取：对于每个滤波器输出的能量值进行离散余弦变换（DCT），提取倒谱系数。DCT可以将时域信号转换为频域信号，提取出主要的频率特征。
   8. 群延迟特征提取：结合群延迟和倒谱系数进行特征提取。在**计算MFCC时，通常只考虑幅度谱**，而**GFCC还利用了群延迟信息，使其能够更好地捕捉语音信号的动态特征**。
   9. 特征归一化：对得到的GFCC系数进行归一化操作，以确保不同语音信号之间的比较具有可靠性。

   以上就是GFCC的原理实现过程。通过使用群延迟和倒谱系数，GFCC能够提取出更具有区分度和表达力的语音特征，对于语音识别和音频处理任务具有较好的效果。

> Gammatone是一种信号处理方法和滤波器设计，用于**模拟人耳的听觉系统对声音的感知**。它基于生物学上已知的耳蜗内部的神经元响应原理。
>
> Gammatone滤波器通过使用**一组相关的带通滤波器**来模拟人耳中的听觉通道。这些**滤波器在频率上相互重叠**，并且其频率响应类似于带谷的**gamma函数**形状，因此得名为Gammatone滤波器。每个滤波器都具有特定的中心频率和带宽，类似于人耳中的听觉通道。
>
> > Gamma函数是数学中的一种特殊函数，通常用符号Γ表示。它是阶乘函数在复数域上的推广。
> >
> > 对于实数x大于0，Gamma函数定义如下：
> >
> > Γ(x) = ∫[0, +∞] t^(x-1) * e^(-t) dt
> >
> > 其中，t^(x-1)表示t的x-1次幂，e是自然对数的底数。
> >
> > Gamma函数还具有以下性质：
> >
> > - 对于正整数n，Γ(n) = (n-1)!
> > - Γ(x+1) = x * Γ(x)，这是Gamma函数的递归关系。
> > - 当x为实数时，Γ(x)在(0, +∞)上连续且无穷多次可导。
> >
> > Gamma函数在数学和物理学中具有广泛的应用。它与组合数学、复变函数、积分等领域有着密切关联。在概率论和统计学中，Gamma函数与贝塔分布、伽马分布等概率分布函数相关联。此外，在物理学中，Gamma函数在量子力学、统计力学等领域也有重要的应用。
>
> <img src="feature engineering.assets/1200px-Gamma_plot_zh.png" alt="1200px-Gamma_plot_zh" style="zoom:50%;" />
>
> Gammatone滤波器广泛用于音频处理、语音识别、音频编码和听觉模型等领域。它可以提取特定频率范围内的声音信息，并模拟人耳对不同频率的敏感性。

- 模板代码：

   ```python
   import numpy as np
   import librosa
   import gammatone.filters as gf
   
   # 加载音频文件
   audio_file = 'path/to/audio.wav'
   y, sr = librosa.load(audio_file)
   
   # 计算GFCC特征
   n_mfcc = 13  # MFCC系数数量
   n_fft = 2048  # FFT长度
   hop_length = 512  # 帧移距离
   
   # 计算Gammatone滤波器组
   num_filters = 40  # 滤波器数量
   low_freq = 80  # 最低频率
   high_freq = 8000  # 最高频率
   filters = gf.make_erb_filters(sr, num_filters, low_freq, high_freq)
   
   # 进行Gammatone滤波
   filtered = gf.filterbank(y, filters)
   
   # 对滤波后信号进行幅度谱计算
   power_spectrum = np.abs(librosa.stft(filtered, n_fft=n_fft, hop_length=hop_length))**2
   
   # 计算MFCC特征
   gfcc = librosa.feature.mfcc(S=librosa.power_to_db(power_spectrum), n_mfcc=n_mfcc)
   
   # 打印GFCC特征
   print('GFCC:')
   print(gfcc)
   ```

请注意，上述代码使用了`gammatone.filters`库来生成Gammatone滤波器组，并使用`filterbank`函数将音频信号进行滤波。然后，通过计算幅度谱并应用MFCC提取步骤，我们可以得到GFCC特征。

请确保你已经安装了`gammatone`和`librosa`库以运行上述代码示例。

## 小波变换

小波变换（Wavelet Transform）是一种用于信号和图像处理的数学工具，具有时频**局部化**的特性。与傅里叶变换（Fourier Transform）相比，小波变换可以提供更好的时域和频域的局部分析能力，能够捕捉信号中的**短时变化**。

小波变换通过将原始信号与一组称为小波基（Wavelet）的函数进行卷积运算来分析信号。小波基是以时间和频率两个尺度进行变化的函数，可以在不同尺度上对信号进行分解和重构。这种尺度变化的特性使得小波变换可以同时提供频域和时域的信息。

小波变换的过程可以分为以下几步：

1. 尺度变换：小波变换使用不同尺度的小波基函数对信号进行分解。通过调整小波基的尺度，可以捕捉信号在不同频率范围内的特征。

2. 位置变换：在每个尺度上，小波基函数在时间轴上进行平移，对信号的不同时间段进行分析。这种平移操作可以提供信号在时间上的局部分析。

3. 卷积运算：在每个尺度和位置上，将原始信号与小波基函数进行卷积运算。卷积运算可以通过计算原始信号与小波基函数的内积来获得对应尺度和位置上的小波系数。

4. 重构：通过对小波系数进行逆变换，可以将信号从小波域重新恢复到原始的时域。

小波变换在信号和图像处理中有多种应用，包括信号的去噪、特征提取、边缘检测、压缩等。由于小波变换具有时频局部化的特性，能够更好地捕捉信号的**瞬时特征**，**`因此在处理非平稳信号和包含短时变化的信号时，小波变换常常比傅里叶变换更为适用`**。

需要注意的是，小波变换是一种多尺度分析方法，有多种小波基函数可供选择，如Haar小波、Daubechies小波、Morlet小波等。不同的小波基函数适用于不同类型的信号和应用场景，选择适合的小波基函数对于获得良好的分析结果是重要的。

小波变换（Wavelet Transform）和梅尔频率倒谱系数（MFCC）是两种常用的特征提取方法，它们在语音信号处理中具有不同的特点和应用。

小波变换的优点是能够提供时频局部化的特征分析，对于捕捉信号的短时变化和非平稳性表现出较好的效果。小波变换可以提供更高分辨率的频域信息，能够较好地分析信号中的瞬时特征和频率变化。因此，在一些需要对信号的时间局部性进行分析的场景中，小波变换可能更适合。

MFCC是一种常用于语音识别的特征提取方法，它主要关注信号的语音信息，对人耳感知较为重要的频率区域有更高的分辨率，而对于较高频率的细节信息进行了抽样。MFCC在语音识别中广泛应用，已经被证明在语音识别任务中具有较好的效果。它对于语音信号的鉴别性和稳定性较强，能够较好地表达语音信号的语音内容。

对于特征效果的比较，很难一概而论哪个方法更好，因为它们在不同的应用场景下可能有不同的表现。选择适合的特征提取方法应该根据具体的任务需求、数据特点和算法模型进行综合考虑。有时候，结合多种特征提取方法可以获得更好的效果。

在实际应用中，通常会根据具体任务的需要进行实验和比较，通过评估指标（如分类准确率、识别率等）来确定哪种特征提取方法更适合。此外，特征选择和模型的选择与调整也是影响特征效果的重要因素。
# 一、概述
图像压缩
Metric
有损压缩图像质量或失真程度一般用峰值信噪比（PSNR）指标来衡量。虽然峰值信噪比不能完全反映人类视觉效果，但是它仍是一个比较流行的量化指标。

## 1.1 传统图像压缩算法：
### 1.1.1 JPEG
JPEG [1 and 2] 是Joint Photographic Experts Group（联合图像专家小组）的缩写，是第一个国际图像压缩标准。JPEG图像压缩算法能够在提供良好的压缩性能的同时，具有比较好的重建质量，被广泛应用于图像、视频处理领域。
JPEG静止图像压缩标准,中端和高端比特速率上的良好的速率畸变特性，但在低比特率范围内，将会出现很明显的方块效应，其质量变得不可接受。
![JPEG 编解码过程](pic/JPEG.jpg)
以baseline JPEG算法(baseline sequential)压缩24位彩色图像为例,压缩步骤如下：
(1). 颜色转换
RGB -> YCbCr

(2). DC电平偏移
作用？DC电平偏移的目的是保证输入图像的采样有近似地集中在零附近的动态范围。

(3). 子采样
YUV420

(4). DCT变换
DCT（DiscreteCosineTransform）是将图像信号在频率域上进行变换，分离出高频和低频信息的处理过程。然后再对图像的高频部分（即图像细节）进行压缩，以达到压缩图像数据的目的。首先将图像划分为多个8x8的矩阵。然后对每一个矩阵作DCT变换。变换后得到一个频率系数矩阵，其中的频率系数都是浮点数。

(5). 量化
对于DCT之后的结果，根据以下的标准量化表进行量化。
![标准亮度量化表](pic/标准亮度量化表.gif)
![标准色差量化表](pic/标准色差量化表.gif)

(6). 编码
编码采用两种机制：
（1）0值的行程长度编码
（2）熵编码
1. Zig-zag ordering
![Zig-zag(https://www.cnblogs.com/tgycoder/p/4991663.html)](pic/zig-zag.png)

2.

### 1.1.2 JPEG2000
JPEG 2000 [8] 是基于小波变换的图像压缩标准，由Joint Photographic Experts Group组织创建和维护。JPEG 2000通常被认为是替代JPEG的下一代图像压缩标准。优势：JPEG2000的压缩比更高，而且不会产生原先的基于离散余弦变换的JPEG标准产生的块状模糊瑕疵。JPEG2000同时支持有损压缩和无损压缩。另外，JPEG2000也支持更复杂的渐进式显示和下载。JPEG2000的失真主要是模糊失真。模糊失真产生的主要原因是在编码过程中高频量一定程度的衰减。传统的JPEG压缩也存在模糊失真的问题。就图像整体压缩性能来说，在低压缩比情形下（比如压缩比小于10：1），传统的JPEG图像质量有可能要比JPEG2000要好。JPEG2000在压缩比比较高的情形下，优势才开始明显。整体来说，和传统的JPEG相比，JPEG2000仍然有很大的技术优势，通常压缩性能大概可以提高20%以上。一般在压缩比达到100：1的情形下，采用JPEG压缩的图像已经严重失真并开始难以识别了，但JPEG2000的图像仍可识别。


基于学习的方法
概述
随着计算能力的发展，深度学习已经在计算机视觉等领域展现出强大的力量，使得通过对大量自然图片进行学习，获得其中的特征表达成为可能。近几年一些学者已经在这方面做出了非常优秀的成果。包括Google、WaveOne、ETH等。
基于学习的方法的基本结构来源于autoencoder，这是一种将图像压缩到特征空间然后再还原为原始图像的结构，其基本结构如下图：
![autoencoder(https://blog.csdn.net/lwq1026/article/details/78581649)](pic/autoencoder.png)
JPEG对空间相关性的利用不够充分，为了充分利用空间相关性。基于学习的方法一般利用
图
[3]中把通过LSTM等方法循环输出码流的方法 称为层进式编码(Pruduce Progessive Codes),这其中典型的代表有google的两篇文章 [4],[5],[6]，部分代码开源在https://github.com/tensorflow/models/tree/master/research/compression

1. 数据集准备
在如今的互联网大数据时代，网络上可以搜集到的图片数据虽然很多，但是原始的无损图像数据非常有限。不过如果从信号恢复的角度来讲，任何格式的图像数据都可以作为无损数据来对网络进行训练。所以很多论文包括谷歌[5]的工作都是直接收集的大量网络图片来制作训练集。
2. 网络结构

3. 损失函数设计

4. RDO

我们目前的工作
后续计划
由于我们目前全图使用的是统一的压缩比例而没有考虑到图像信息量在不同区域的区别，所以下一步考虑在不同的区域使用不同的量化级别进行压缩，这方面也已经有了一些相关的工作包括google [6] 和 港科大[7].

# references
[1] https://en.wikipedia.org/wiki/JPEG
[2] https://baike.baidu.com/item/JPEG
[3] Learning to inpaint for Image Compression
[4] Toderici, G., O'Malley, S. M., Hwang, S. J., Vincent, D., Minnen, D., Baluja, S., ... & Sukthankar, R. (2015). Variable rate image compression with recurrent neural networks. arXiv preprint arXiv:1511.06085.
[5] Toderici, G., Vincent, D., Johnston, N., Hwang, S. J., Minnen, D., Shor, J., & Covell, M. (2017, July). Full resolution image compression with recurrent neural networks. In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on (pp. 5435-5443). IEEE.
[6] Johnston, N., Vincent, D., Minnen, D., Covell, M., Singh, S., Chinen, T., ... & Toderici, G. (2017). Improved lossy image compression with priming and spatially adaptive bit rates for recurrent networks. arXiv preprint arXiv:1703.10114.
[7] Li, M., Zuo, W., Gu, S., Zhao, D., & Zhang, D. (2017). Learning convolutional networks for content-weighted image compression. arXiv preprint arXiv:1703.10553.
[8] https://baike.baidu.com/item/JPEG%202000/8097196?fromtitle=jpeg2000&fromid=5452998

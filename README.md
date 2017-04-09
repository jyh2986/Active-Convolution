# Active Convolution

This repository contains the implementation for the paper [Active Convolution: Learning the Shape of Convolution for Image Classification](http://arxiv.org/abs/1703.09076). 

The code is based on [Caffe](https://github.com/BVLC/caffe) and [cuDNN](https://developer.nvidia.com/cuDNN)(v5)


## Abstract

<table>
<tr>
<td width = 70%>

In recent years, deep learning has achieved great success in many computer vision applications. Convolutional neural networks (CNNs) have lately emerged as a major approach to image classification. Most research on CNNs thus far has focused on developing architectures such as the Inception and residual networks. The convolution layer is the core of the CNN, but few studies have addressed the convolution unit itself. <b>In this paper, we introduce a convolution unit called the active convolution unit (ACU). A new convolution has no fixed shape, because of which we can define any form of convolution. Its shape can be learned through backpropagation during training.</b> Our proposed unit has a few advantages. First, the ACU is a generalization of convolution; it can define not only all conventional convolutions, but also convolutions with fractional pixel coordinates. We can freely change the shape of the convolution, which provides greater freedom to form CNN structures. Second, the shape of the convolution is learned while training and there is no need to tune it by hand. Third, the ACU can learn better than a conventional unit, where we obtained the improvement simply by changing the conventional convolution to an ACU. We tested our proposed method on plain and residual networks, and the results showed significant improvement using our method on various datasets and architectures in comparison with the baseline. 
</td>
<td>
<img src = 'https://cloud.githubusercontent.com/assets/25662811/24835483/766d7aee-1d3e-11e7-8b62-019df7c48d82.png'>
</td>
</tr>
</table>


## Testing Code
You can validate backpropagation using test code.
Because it is not differentiable on lattice points, you should not use integer point position when you are testing code.
It is simply possible to define "TEST_ACONV_FAST_ENV" macro in <i>aconv_fast_layer.hpp</i>

1. Define "TEST_ACONV_FAST_ENV" macro in aconv_fast_layer.hpp
2. \> make test
3. \> ./build/test/test_aconv_fast_layer.testbin

You should pass all tests.
Before the start, <b>don't forget to undefine TEST_ACONV_FAST_ENV macro and make again.</b>

## Usage

ACU has 4 parameters(weight, bias, x-positions, y-positions of synapse).
Even though you don't use bias term, the order will not be changed.

Please refer [deploy file](https://github.com/jyh2986/Active-Convolution/blob/publish_ACU/models/ACU/[plain]deploy.prototxt) in models/ACU

If you want define arbitary shape of convolution,

1. use non SQUARE type in aconv_param 
2. define number of synapse using kernel_h, kernel_w parameter in convolution_param


In example, if you want define cross-shaped convolution with 4 synapses, you can use like belows.

```
...
aconv_param{   type: CIRCLE }
convolution_param {    num_output: 48    kernel_h: 1    kernel_w: 4    stride: 1 }
...
```

When you use user-defined shape of convolution, you'd better edit <i>aconv_fast_layer.cpp</i> directly to define initial position of synapses. 


## Example

This is the result of plain ACU network, and there an example in [models/ACU](https://github.com/jyh2986/Active-Convolution/blob/publish_ACU/models/ACU) of CIFAR-10

| Network  | CIFAR-10(\%) | CIFAR-100(\%) 
|:-------|:-----:|:-------:|
| baseline | 8.01 | 27.85 |
| ACU      | 7.33 | 27.11 |
| Improvement | <b>+0.68</b> | <b>+0.74</b> |


This is changes of the positions over iterations.

<img src=https://github.com/jyh2986/Active-Convolution/blob/publish_ACU/models/ACU/plain.gif width=30%>

You can draw learned position by using [ipython script](https://github.com/jyh2986/Active-Convolution/blob/publish_ACU/models/ACU/net_view.ipynb).

# torcher

torcher是为pytorch训练模型提供一个类似keras和sklearn中fit方法类似的接口，使用步骤如下
1. 准备好要进行评估的模型（一个`torch.nn.Module`的实例），一个loss函数以及一系列评价指标；
1. 声明一个Torcher对象（可通过`from torcher import Torcher`导入定义）；
3. 调用fit方法进行拟合。

模块实现见torcher文件夹，test.py提供了一个基于MNIST的简单实例。

## loss函数

任意的接受两个`tensor`并返回一个标量`tensor`的函数都可以被当作loss函数，可表示为
```
f(y_pred,y_true) -> loss
```
其中`y_pred`表示一个batch的预测结果，`y_true`表示对应的label。

## metric函数
任意的接受两个`tensor`并返回一个**标量**的函数都可以被当作loss函数，可表示为
```
f(y_pred,y_true) -> loss
```
其中`y_pred`表示一个batch的预测结果，`y_true`表示对应的label。注意该函数与loss不同的是，其返回值为numpy类型，这允许你在计算指标时使用任意不可求导的方法。

## transfrom
transform函数用来对输入进行一些与处理，transform对象应当是可调用的。特征在transform函数中进行的变换不会被计算梯度，这允许你在GPU上使用一些函数来加速模型，例如在GPU上进行FFT变换。transform对象应当同model对象一样被放置在GPU上。

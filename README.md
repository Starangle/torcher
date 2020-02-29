# torcher

torcher是为pytorch训练模型提供一个类似keras和sklearn中fit方法类似的接口，使用步骤如下
1. 准备好要进行评估的模型（一个`torch.nn.Module`的实例），一个loss函数以及一系列评价指标；
1. 声明一个Torcher对象（可通过`from torcher import Torcher`导入定义）；
3. 调用fit方法进行拟合。
model对象应当已经被放置到GPU上。

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
f(y_pred,y_true) -> metric
```
其中`y_pred`表示一个batch的预测结果，`y_true`表示对应的label。注意该函数与loss不同的是，其返回值为numpy类型，这允许你在计算指标时使用任意无法计算梯度的方法。

## transfrom对象
在处理数据时，如果希望能够利用GPU进行加速，可以指定transform对象，transform对象应当是一个`torch.nn.Module`的子类，在Torcher中，transform对象中进行的操作不会被计算梯度。transform对象应当独立于`model`对象，因为pytorch不允许在模型中存储可能用到的一些函数，例如torchaudio中的FFT变换。

## optimizer对象
`opti`参数接受形如
```
lambda x:torch.optim.Adam(x,lr=0.01)
```
的函数对象，其中x会在Torcher对象中为其赋值，其余超参数可以在自己进行设定。

# 0.0.5更新
* 修复了log_file在当前目录下因无法创建文件夹而导致的错误
* 现在transform由用户的类型为用户编写的`torch.nn.Module`的子类
* model和transform对象中数据在CPU和GPU之间的迁移应当由用户决定，以便于构建在多个GPU上并行的模型
* 支持DataLoader输出的某个batch为`None,None`的操作，这个批次被直接跳过
* 增加学习率衰减的回调函数，支持按照loss衰减和按照训练时间衰减两种模式，通过`method`来指定，关键字分别为`based_on_loss`和`based_on_epoch`
    * loss:当连续`freq`个epoch中没有出现loss下降，则将学习率乘上`decay`的数值
    * 按时间衰减：每经过`freq`个epoch，将学习率乘上`decay`的数值
* 修复`valid_data=None`时的Bug
* 增加名为Checkpoint的回调函数，自动保存在验证集上具有最小的loss的模型
    *可选择保存具有最高指标的模型
* 增加eval方法，可以按照设定好的loss和metrics进行验证并报告结果

# 0.0.4更新
* 现在Torcher支持将日志写入到文件中，相比与使用重定向，这可以避免一些提示性的输出（例如进度条）。可以通过`fit`函数的`log_file`参数来指定写入的文件。

# torcher

torcher是为pytorch训练模型提供一个类似keras和sklearn中fit方法类似的接口，使用步骤如下
1. 准备好要进行评估的模型（一个`torch.nn.Module`的实例），一个loss函数（返回一个标量`tensor`）以及一系列评价指标（每个函数返回一个标量）；
1. 声明一个Torcher对象（可通过`from torcher import Torcher`导入定义）；
3. 调用fit方法进行拟合。

模块实现见torcher文件夹，test.py提供了一个基于MNIST的简单实例。

目前不支持更换优化器，使用默认的Adam优化器。
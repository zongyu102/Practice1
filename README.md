# Practice1
## Introduction
##### 该项目是利用pytorch框架实现对数据集cifar-10的分类,模型的网络结构借鉴了vgg网络。  
##### cifar-10数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批  
##### 10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。  
## File
##### main.py文件用于实现模型,并生成中间数据文件  
##### drawgradio.py文件利用gradio库实现数据集可视化，分类结果可视化以及训练过程Loss和Acc的可视化。  
##### drawAcc.py文件在结合过程数据文件绘制Acc散点图  
##### drawAvgLoss.py文件用于绘制每次epoch的平均loss,修改文件路径和标题后也可绘制训练过程的loss和acc  
##### Data/trainData中存放了不同batch size以及不同epoch下的训练数据(包括训练过程的loss和acc变化以及每次epoch的平均loss变化)  
##### data/accData中存放了不同batch size、learn rate以及epoch下的测试集acc变化  
##### image中存放了从官方下载的cifar-10数据集

import torch
from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torchvision
import torch.nn.functional as function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import numpy as np
from PIL import Image
from time import time
import matplotlib.pyplot as plt
#print(torch.__version__)
#print(torch.cuda.is_available())
#print(isinstance(2,torch.ByteTensor))
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
epoch = 40
trans = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#训练集
traindata = torchvision.datasets.CIFAR10(
    root='./image',
    train=True,
    transform=trans,
    download=True
)

#测试集
testdata = torchvision.datasets.CIFAR10(
    root='./image',
    train=False,
    transform=trans,
    download=True

)

def tensor2im(input_image, imtype = np.uint8):#将图片从tensor格式转化为image的numpy格式
    if not isinstance(input_image, np.ndarray):#图片不是np格式
        if isinstance(input_image, torch.Tensor):#图片是tensor格式
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()#numpy不能读取CUDA tensor需要转化为CPU tensor
        if image_numpy.shape[0] == 1:#处理灰度图转换为RGB
            image_numpy = np.title(image_numpy, (3, 1, 1))
        for i in range(0, 3):#反标准化
            image_numpy[i] = image_numpy[i] * 0.5 + 0.5
        image_numpy = image_numpy * 255#将数值从[0, 1]转换为[0, 255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))#从(C,H,W)->(H,W,C)
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


#数据加载
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)
print('testdata', len(testloader))
#10组照片分类
imageclass = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#网络模型,采用CNN
#学习率
LR = 0.0005

#网络结构
class CNet(torch.nn.Module):
    def __init__(self):
      super(CNet, self).__init__()

      self.conv1 = torch.nn.Sequential(
          torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), #(32-3+2)/1+1=32, 3*32*32->16*32*32
          torch.nn.BatchNorm2d(16),#将神经元输入拉回到标准正态分布
          torch.nn.ReLU(),

          torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(16),
          torch.nn.ReLU(),

          #torch.nn.MaxPool2d(2, 2)#(32+2*0-1*(2-1)-1)/2+1=16, 16*32*32->16*16*16,池化层提取显著特征
      )
      self.conv2 = torch.nn.Sequential(
          torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),#16*16*16->32*16*16(3层)  或 (5层）32*32*32
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),

          torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),#32*16*16->32*16*16（3层），（5层）32*32*32->32*32*32
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),

          torch.nn.MaxPool2d(2, 2)#32*16*16->32*8*8(3层)，（5层）32*16*16
      )
      self.conv3 = torch.nn.Sequential(
          torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#32*8*8->64*8*8（3层），（5层）32*16*16->64*16*16
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),

          torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          #torch.nn.MaxPool2d(2, 2)#64*8*8->64*4*4
      )
      self.conv4 = torch.nn.Sequential(
          torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64*16*16->128*16*16
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),

          torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),

          torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),

          torch.nn.MaxPool2d(2, 2)#128*16*16->128*8*8
      )
      self.conv5 = torch.nn.Sequential(
          torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  #128*8*8->256*8*8
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),

          torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),

          torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),

          torch.nn.MaxPool2d(2, 2)  # 256*8*8->256*4*4
      )
      self.func = torch.nn.Sequential(
          torch.nn.Linear(256*4*4, 32),
          torch.nn.ReLU(),
          #torch.nn.Dropout(0.2),

          torch.nn.Linear(32, 10)
      )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 256*4*4)
        x = self.func(x)
        return x

net = CNet()
net.cuda()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

train_loss = []
train_acc = []
train_allloss = []
#训练
def train(model, loss_f, optimizer, traindata, epochs, log_interval=50):
    print('Train')
    for tepoch in range(epochs):
        all_loss = 0
        correct = 0
        num = 0
        for step, (batch_x, batch_y) in enumerate(traindata):
            #print(step)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            output = model(batch_x)
            _, predicted = torch.max(output.data, 1)
            num += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            optimizer.zero_grad()#梯度归零

            loss = loss_f(output, batch_y)#计算损失
            train_loss.append(loss.item())
            loss.backward()#反向传播计算每个参数的梯度值
            optimizer.step()#通过梯度下降进行参数更新

            all_loss = all_loss + loss
            print('%d loss: %.4f' % (step + 1, loss))
        #print('%d loss: %.4f' % (tepoch + 1, all_loss))
        acc = correct / num
        all_loss_avg = all_loss / num
        train_allloss.append(all_loss_avg)
        train_acc.append(acc)
        print('%d acc: %.4f' % (tepoch + 1, acc))
    print('train finish')

original_list = []
predicted_list = []
image_tensor_list = []
image_numpy_list = []
#测试
def test(model, testdata):
    print('Test')

    real = 0
    total = 0

    with torch.no_grad():#反向传播时不用求导，节省空间
        for x, y in testdata:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            _, predicted = torch.max(output.data, 1)#获取得分最高的类即从10个类中选出分数最高的类
            total += y.size(0)
            real += (predicted == y).sum().item()
            for i in range(0, y.size(0)):
                olabel = y.data
                plabel = predicted.data
                original_list.append(olabel[i])
                predicted_list.append(plabel[i])
                image_tensor_list.append(x[i])
            #print('比对结果',predicted, y)

    hit = 100 * real / total
    print("Accuracy of the network is: %.4f %%" % hit)
    return hit

train(net, loss_fun, optimizer, trainloader, epochs=epoch)
test(net, testloader)


for i in range(0, len(image_tensor_list)):
    image_numpy = tensor2im(image_tensor_list[i])
    image_numpy_list.append(image_numpy)
print(len(image_numpy_list))
#(image_numpy_list[0])
#title_true = 'true='+str(original_list[0])
#title_prediction = ', prediction'+str(predicted_list[0])
#title = title_true + title_prediction
#plt.title(title)
#plt.xticks([])
#plt.yticks([])
#plt.savefig("./image0/")
#plt.show()

#torch.save(net.state_dict, './model_save/cnet.pt')
with open("./loss_acc/train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss))

with open("./loss_acc/train_acc.txt", 'w') as train_ac:
    train_ac.write(str(train_acc))

with open("./loss_acc/train_allloss.txt", 'w') as train_allloss:
     train_allloss.write(str(train_allloss))

np.save(file="./predicted/image.npy", arr=image_numpy_list)
np.save(file="./predicted/original_label.npy", arr=original_list)
np.save(file="./predicted/predicted_label.npy", arr=predicted_list)


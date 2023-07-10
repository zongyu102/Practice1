import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)

def drawAvgLoss(path, savepath):
    y_acc = data_read(path)
    #x_batch = [32, 64, 128, 256, 512]#x轴显示的刻度
    #x = [1, 2, 3, 4, 5]#用于等间距分割
    x_learn = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    x = [1, 2, 3, 4, 5, 6]
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.xlabel('batchsize')
    plt.xlabel('learn rate')
    plt.ylabel('acc')
    #plt.plot(x_new, y_smooth, color='blue', linestyle="solid", label="test acc")
    plt.scatter(x, y_acc, color='blue', alpha=0.5)
    #plt.xticks(x, x_batch)
    plt.xticks(x, x_learn)
    plt.title('Test Acc')
    plt.savefig(savepath)
    plt.show()

if __name__ == "__main__":
    batch_acc = r"./Data/accData/batch64/epoch40/learn_acc.txt"
    savepath = "./Data/accDataImage/batch64/epoch40/learn_acc.png"
    drawAvgLoss(batch_acc, savepath)
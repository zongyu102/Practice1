import numpy as np
import matplotlib.pyplot as plt


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


def drawAvgLoss(path, savepath):
    y_avgloss = data_read(path)
    x_avgloss = range(len(y_avgloss))
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('epoch')
    plt.ylabel('avg loss')
    plt.plot(x_avgloss, y_avgloss, color='blue', linestyle="solid", label="train avg loss")
    plt.title('Train Avg Loss')
    plt.savefig(savepath)
    plt.show()


if __name__ == "__main__":
    train_avg_loss_path = r"./trainData/batch=64/LR=0.0005/epoch40/train_epoch_avg_loss.txt"
    savepath = "./trainDataImage/batch64/LR0.0005/epoch40/train_avg_loss.png"
    drawAvgLoss(train_avg_loss_path, savepath)


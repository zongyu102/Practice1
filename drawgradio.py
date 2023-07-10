import gradio as gr
import pickle
import glob
import numpy as np
import os
from PIL import Image

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

train_list = glob.glob("./image/cifar-10-batches-py/data_batch_*")
test_list = glob.glob("./image/cifar-10-batches-py/test_batch")
num = 0

label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
image_list = []
label_list = []
p_original_image = np.load("./gradioView/predicted/image.npy")
p_original_label = np.load("./gradioView/predicted/original_label.npy")
p_predicted_label = np.load("./gradioView/predicted/predicted_label.npy")

for l in train_list:
    l_dict = unpickle(l)
    #print(l_dict.keys())
    #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    for im_idx, im_data in enumerate(l_dict[b"data"]):
        # 遍历字典中 data 这个维度
        # enumerate（）函数表示将列表、字符串等可遍历的数据对象组成一个索引序列
        # print(im_idx)
        # print(im_data)
        im_name = l_dict[b"filenames"][im_idx]
        im_label = l_dict[b"labels"][im_idx]
        #print(im_name, im_label, im_data)

        im_lable_name = label_name[im_label]#Cifar10 里面的label 是0 1 2 3 4 故转化为 dog ship
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, [1, 2, 0])# 对空间矩阵的进行转置
        image_list.append(im_data)
        label_list.append(im_lable_name)

for l in test_list:
    l_dict = unpickle(l)
    #print(l_dict.keys())
    #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    for im_idx, im_data in enumerate(l_dict[b"data"]):
        # 遍历字典中 data 这个维度
        # enumerate（）函数表示将列表、字符串等可遍历的数据对象组成一个索引序列
        # print(im_idx)
        # print(im_data)
        im_name = l_dict[b"filenames"][im_idx]
        im_label = l_dict[b"labels"][im_idx]
        #print(im_name, im_label, im_data)

        im_lable_name = label_name[im_label]#Cifar10 里面的label 是0 1 2 3 4 故转化为 dog ship
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, [1, 2, 0])# 对空间矩阵的进行转置
        image_list.append(im_data)
        label_list.append(im_lable_name)


def greet(id):
    id = int(id)
    tim = Image.fromarray(image_list[id])
    #tim = tim.resize((1024, 1024))
    tlabel = label_list[id]
    #tim.show()
    return tim, tlabel

def greet2(id2):
    id = int(id2)
    tim = Image.fromarray(p_original_image[id])
    olabel_numb = p_original_label[id]
    plabel_numb = p_predicted_label[id]

    return tim, label_name[olabel_numb], label_name[plabel_numb]


with gr.Blocks() as demo:
    with gr.Tab("原始数据集"):
        with gr.Row():
            with gr.Column(scale=3, min_width=10):
                id = gr.Textbox(label="Image Id[0,59999)", min_width=10)
                label = gr.Label(label="Class", scale=1)
            with gr.Column(scale=1, min_width=10):
                image = gr.Image(label="Image",shape=(32, 32), scale=4)
        greet_btn = gr.Button("Greet", scale=1, min_width=1)
        greet_btn.click(fn=greet, inputs=id, outputs=[image, label], api_name="greet")

    with gr.Tab("分类结果"):
        with gr.Row():
            with gr.Column(scale=3, min_width=10):
                id2 = gr.Textbox(label="Image Id[0,9999)", min_width=10)
                with gr.Row():
                    label1 = gr.Label(label="Original Class", scale=1)
                    label2 = gr.Label(label="Predicted Class", scale=1)
            with gr.Column(scale=1, min_width=10):
                image2 = gr.Image(label="Image",shape=(32, 32), scale=4)
        greet_btn2 = gr.Button("Greet", scale=1, min_width=1)
        greet_btn2.click(fn=greet2, inputs=id2, outputs=[image2, label1, label2], api_name="greet")
    with gr.Tab("训练过程loss和accuracy"):
        with gr.Row():
            gr.Image("./gradioView/train_acc_loss/train_loss.png")
            gr.Image("./gradioView/train_acc_loss/train_acc.png")

demo.launch()
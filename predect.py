# -*- coding: UTF-8 -*-
"""
@Project ：zzytgitee
@File ：Test_20210806.py
@Author ：正途皆是道
@Date ：21-8-6 上午9:08
"""
import time
import csv
import os
import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle import nn
from paddle.vision import models
import numpy as np
from PIL import Image
from paddle.vision import transforms

from visualdl import LogWriter
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=0, help="model filename")
    parser.add_argument("--gpu", type=bool, default=False, help="model filename")
    return parser.parse_args()
# paddle.device.set_device("cpu") #在gpu环境下使用cpu


rgb_mean = (67.0, 33.0, 11.0)  # 训练图片的均值
rgb_std = (75.0, 40.0, 19.0)  # 训练图片的方差
aug_p = 0.3  # 每个数据增强操作发生的概率
num_classes=3
label_text = ''
train_dir = ''
test_dir = 'data/data117383/train_img/Test'
INPUT_SIZE = (1000, 1000)
BATCH_SIZE = 1



class CatNn(nn.Layer):
    def __init__(self):
        """模型结构堆叠"""
        super(CatNn, self).__init__()
        # 参数with_pool产生The fisrt matrix width should be same as second matrix height,
        # but received fisrt matrix width
        # self.base = models.MobileNetV2(num_classes=12)
        # self.base = models.ResNet(BasicBlock, 18)
        self.base = models.ResNet(models.resnet.BottleneckBlock, depth=50, num_classes=num_classes, with_pool=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.tanhshrink = nn.Tanhshrink()

    def forward(self, x):
        x = self.base(x)
        x = self.softmax(x)
        # x = self.tanhshrink(x)
        # x = self.tanh(x)
        return x


class LoadNnData(Dataset):
    def __init__(self, imgs_path=[], label_list=[], hw_scale=(1080, 1920), dataaug=None, model='train'):
        """载入图像分类数据"""
        super(LoadNnData, self).__init__()
        self.model = model
        self.imgs_path = imgs_path
        self.labels = label_list
        self.hw = hw_scale
        self.num_samples = len(imgs_path)
        '''数据增强'''
        self.dataaug = dataaug
        self.aug_p = aug_p

    def __getitem__(self, item):
        if self.model == 'train':  # 仅训练时进行数据增强
            # image = process_image(self.imgs_path[item], self.model)
            image = self.dataaug.augop(self.imgs_path[item], self.aug_p)  # paddle提供的数据增强库
        else:  # 预测和评估时不需要增强,普通归一化即可
            image = self.img_to_chw_rgb(self.imgs_path[item])

        # image = paddle.to_tensor(image)  # CHW
        if len(self.labels) == 0:  # 预测时不提供label信息
            return image, self.imgs_path[item]

        if self.model == 'eval':  # 评估数据
            return image, self.labels[item], self.imgs_path[item]

        # label = paddle.to_tensor(self.labels[item])  # 参考1,2,3,4,5,loss,accuracy的输入维度[intN]
        return image, np.array([self.labels[item]])  # 训练数据 data,label

    def __len__(self):
        return self.num_samples

    def img_to_chw_rgb(self, img_path):
        pil_img = Image.open(img_path)
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        pil_img = transforms.resize(pil_img, INPUT_SIZE, interpolation='bilinear')  # 将输入数据调整为指定大小。
        pil_img = transforms.normalize(pil_img, rgb_mean, rgb_std, data_format='HWC')  # 归一化
        pil_img = transforms.Transpose(order=(2, 0, 1))(pil_img)  # 将输入的图像数据更改为目标格式 HWC -> CHW
        pil_img = pil_img.astype('float32')
        return pil_img


class ReadData(object):
    def __init__(self):
        """读取图像信息"""
        super(ReadData, self).__init__()

    def get_imgpath_label(self, label_file, img_dir, scale=1):
        # 获取图像和标签,生成交叉验证集8:2
        """返回图片绝对路径和对应的标签"""
        labels, img_paths = [], []
        with open(label_file, 'r') as f_dir:
            lines = f_dir.readlines()
        np.random.shuffle(lines)  # 打乱顺序
        for line in lines:
            line = line.strip()
            line_list = line.split('\t')
            img_paths.append(os.path.join(img_dir, line_list[0]))  # 绝对路径
            labels.append(int(line_list[1]))
        if scale == 1:  # 所有数据均为训练数据
            return labels, img_paths
        split_flag = int(len(labels) * scale)
        train_lab, eval_lab = labels[:split_flag], labels[split_flag:]
        train_img, eval_img = img_paths[:split_flag], img_paths[split_flag:]
        return train_lab, train_img, eval_lab, eval_img

    def get_dir_img_paths(self, img_dir):
        # 返回绝对路径
        return [os.path.join(img_dir, name) for name in os.listdir(img_dir)]


class app():
    def __init__(self):
        """训练、预测、评估"""
        super(app).__init__()

    def predect(self, model_id):
        model = CatNn()
        model_state_dict = paddle.load('./save_model/train/train_{}.pdparams'.format(model_id))
        model.load_dict(model_state_dict)
        model.eval()  # 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False

        predata = ReadData()
        img_paths = predata.get_dir_img_paths(test_dir)

        test_dataset = LoadNnData(imgs_path=img_paths, hw_scale=INPUT_SIZE, model='pre')
        loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
        os.system('rm -rf Classification_Results.csv')
        with open('Classification_Results.csv', 'w+', encoding='utf-8-sig', newline='') as csf:
            writer = csv.writer(csf)
            writer.writerow(['data','non','early','mid_advanced'])
            for batch_id, data in enumerate(loader_test()):
                inputs, paths = data
                predicts = model(inputs)
                result_zero = [0,0,0]
                for i, path in enumerate(paths):
                    name = path.split('/')[-1].replace('.jpg','')
                    predict = np.argmax(predicts.numpy()[i])
                    result_zero[predict] = 1
                    non,early,mid_advanced = result_zero
                    writer.writerow([name, non,early,mid_advanced])
                    print(batch_id * BATCH_SIZE + i, name, non,early,mid_advanced)
            # 这张图片读不起，也只能随便写了
            # writer.writerow(['Qt29gPjYZwv3B6RJh5yiTWXrVImue1FH.jpg', 0])



if __name__ == '__main__':
    args = parse_args()
    print('args.gpu：{}\r\n'.format(args.gpu),
          '\rargs.model_id：{}'.format(args.model_id))
    if args.gpu==False:
        paddle.device.set_device("cpu") #在gpu环境下使用cpu
    app().predect(args.model_id)

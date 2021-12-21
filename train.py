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
import random
import numpy as np
from PIL import Image
from paddle.vision import transforms
from paddle.static import InputSpec
from paddle.jit import to_static
from visualdl import LogWriter
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=None, help="model filename")
    parser.add_argument("--gpu", type=bool, default=True, help="model filename")
    return parser.parse_args()
args = parse_args()
# 二次训练导入参数模型
continue_id = args.model_id  # 首次次训练continue_id = None
# paddle.device.set_device("cpu") #在gpu环境下使用cpu

os.system('rm -rf ./log/scalar/*')  # 每次新训练时都删除旧的可视化log
os.system('rm -rf ./log/dygraph/*')  # 每次新训练时都删除旧的可视化log
train_log = LogWriter(logdir="./log/scalar")
train_log.add_hparams({'learning rate': 0.0001, 'batch size': 2, 'optimizer': 'Adam'}, ['train/loss', 'train/acc'])
train_dygraph = LogWriter(logdir='./log/dygraph')
# visualdl --logdir ./log --port 8080
# http://127.0.0.1:8080
rgb_mean = (67.0, 33.0, 11.0)  # 训练图片的均值
rgb_std = (75.0, 40.0, 19.0)  # 训练图片的方差
aug_p = 0.3  # 每个数据增强操作发生的概率
num_classes=3
label_text = 'data/data117383/train_img/ConcatenateTrain.txt'
train_dir = 'data/data117383/train_img'
test_dir = 'data/data117383/test_img/Test'
INPUT_SIZE = (1000, 1000)
BATCH_SIZE = 14



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


class PaddleAug():
    def __init__(self, mean, std):
        """数据增强"""
        # mean=[127.5, 127.5, 127.5]或者 11.0
        # std=[127.5, 127.5, 127.5] 或者 11.0 每个通道独立使用或者统一一个值
        self.mean = mean
        self.std = std

    def augop(self, pil_img, p):
        """如果给的是路径，则需要读取图片"""
        if type(pil_img) == str:
            pil_img = Image.open(pil_img)
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        """亮度、对比度、饱和度和色调"""
        if random.random() < p:
            pil_img = transforms.adjust_brightness(pil_img, random.randint(5, 15) / 10.0)  # 图片随机增强0.5到1.5倍亮度
        if random.random() < p:
            pil_img = transforms.adjust_contrast(pil_img, random.randint(5, 15) / 10.0)  # 图片随机增强0.5到1.5倍对比度
        if random.random() < p:
            pil_img = transforms.adjust_hue(pil_img, random.randint(-1, 1) / 10.0)  # 图像的色调通道的偏移量最大范围[-0.5,0.5]
        if random.random() < p:
            pil_img = transforms.SaturationTransform(random.randint(0, 2) / 10.0)(pil_img)  # 调整图像的饱和度
        if random.random() < p:
            pil_img = transforms.HueTransform(random.randint(0, 2) / 10.0)(pil_img)  # 调整图像的色调。
        # pil_img = transforms.ColorJitter(random.randint(0, 3) / 10.0, random.randint(0, 3) / 10.0,
        #                                  random.randint(0, 3) / 10.0, random.randint(0, 3) / 10.0,
        #                                  keys=None)(pil_img)  # 随机调整图像的亮度，对比度，饱和度和色调。

        """裁剪，resize"""
        if random.random() < p:
            pil_img = transforms.resize(pil_img, (500, 500), interpolation='bilinear')  # 将输入数据调整为指定大小。
            pil_img = transforms.RandomCrop(450)(pil_img)  # 在随机位置裁剪输入的图像，先将图像进行resize，保证尽量多的保留信息
        """图像翻转"""
        # pil_img = transforms.hflip(pil_img)  # 对输入图像进行水平翻转。
        # pil_img = transforms.vflip(pil_img)  # 对输入图像进行垂直方向翻转。
        pil_img = transforms.RandomHorizontalFlip(p)(pil_img)  # 基于概率来执行图片的水平翻转。
        pil_img = transforms.RandomVerticalFlip(p)(pil_img)  # 基于概率来执行图片的垂直翻转。
        """图像旋转"""
        if random.random() < p:
            pil_img = transforms.RandomRotation(90)(pil_img)  # 依据参数随机产生一个角度对图像进行旋转。
        if random.random() < p:
            pil_img = transforms.rotate(pil_img, 45)  # 按角度旋转图像
        """图像归一化"""
        # if random.random() < p:
        #     to_rgb = random.choice([False, True])
        #     pil_img = transforms.normalize(pil_img, self.mean, self.std, data_format='HWC',to_rgb=to_rgb) # 图像归一化处理
        pil_img = transforms.normalize(pil_img, self.mean, self.std, data_format='HWC')  # 图像归一化处理,to_rgb=to_rgb
        """调整图片大小"""
        pil_img = transforms.resize(pil_img, INPUT_SIZE, interpolation='bilinear')  # 将输入数据调整为指定大小。
        """图像维度置换,展示时注释掉本操作"""
        pil_img = transforms.Transpose(order=(2, 0, 1))(pil_img)  # 将输入的图像数据更改为目标格式 HWC -> CHW

        return pil_img.astype('float32')


class app():
    def __init__(self):
        """训练、预测、评估"""
        super(app).__init__()

    def train(self):
        def del_models():
            # 删除模型，保留新的模型
            models_ctime = []
            model_dir = 'save_model/train'
            for name in os.listdir(model_dir):
                model_path = os.path.join(model_dir, name)
                ctime = os.path.getctime(model_path)  # 创建时间
                models_ctime.append([model_path, ctime])
            models_ctime.sort(key=lambda x: x[1])
            for model_ctime in models_ctime[0:-2]:
                # print('删除模型数据：', model_ctime[0])
                os.system('rm -rf {}'.format(model_ctime[0]))

        dataaug = PaddleAug(rgb_mean, rgb_std)
        predata = ReadData()
        labels, imgs_path = predata.get_imgpath_label(label_text, train_dir)
        train_dataset = LoadNnData(imgs_path=imgs_path, label_list=labels, hw_scale=INPUT_SIZE, dataaug=dataaug,
                                   model='train')
        loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)

        model = CatNn()
        model.train()  # 启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
        opt_function = paddle.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, parameters=model.parameters())
        loss_function = nn.CrossEntropyLoss()  # 多label，soft_label=True
        if continue_id != None:  # 二次训练，读取保存的参数
            layer_state_dict = paddle.load('./save_model/train/train_{}.pdparams'.format(continue_id))
            opt_state_dict = paddle.load('./save_model/train/train_{}.pdopt'.format(continue_id))
            model.set_state_dict(layer_state_dict)
            opt_function.set_state_dict(opt_state_dict)

        for epoch_id in range(200):
            # 开始训练
            tp_num, in_num = 0., 0.
            epoch_t_start = time.time()
            for batch_id, (image, label) in enumerate(loader_train()):
                epoch_b_start = time.time()
                out = model(image)
                loss = loss_function(out, label)
                acc = paddle.metric.accuracy(out, label)
                epoch_b_end = time.time()

                for i in range(BATCH_SIZE):  # 计算全局正确率
                    try:
                        predict = np.argmax(out.numpy()[i])
                        gt = label.numpy()[i]
                        if gt == predict:
                            tp_num += 1
                        in_num += 1
                    except:
                        '''不是一个完整的BATCH_SIZE'''

                if batch_id % 5 == 0:
                    print(
                        "train: epoch:{:>3d}, epoch_time:{:>6.3f}, batch_id:{:>2d}, batch_time:{:>5.3f}, loss is:{:>5.3f}, accuracy is:{:>5.3f}, global accuracy:{:>5.3f}".format(
                            epoch_id, epoch_b_end - epoch_t_start, batch_id, epoch_b_end - epoch_b_start,
                            loss.numpy()[0], acc.numpy()[0], tp_num / in_num))

                """写入visualDL"""
                log_step = epoch_id * BATCH_SIZE + batch_id
                # 向记录器添加一个tag为`acc`的数据
                train_log.add_scalar(tag="train/acc", step=log_step, value=acc.numpy()[0])
                # 向记录器添加一个tag为`loss`的数据
                train_log.add_scalar(tag="train/loss", step=log_step, value=loss.numpy()[0])
                """图片可视化"""
                train_dygraph.add_scalar(tag="dygraph/loss", step=log_step, value=loss.numpy()[0])
                train_dygraph.add_scalar(tag="dygraph/acc", step=log_step, value=acc.numpy()[0])
                # 添加一个图片数据
                log_img = transforms.Transpose(order=(1, 2, 0))(image[0]).numpy().astype('uint8')
                train_dygraph.add_image(tag="dygraph/number_graph", step=log_step, img=log_img)
                # 记录训练参数
                # vdl_train.add_histogram(tag=mnist.parameters()[0].name, values=x,step=step)
                # 权衡精度与召回率之间的平衡关系,对2分类进行计算
                # train_dygraph.add_pr_curve(tag='dygraph/pr_curve', labels=labels, predictions=out, step=log_step,num_thresholds=5)

                loss.backward()
                opt_function.step()
                opt_function.clear_grad()
            if epoch_id % 2 == 0:
                # 增量训练模型参数保存*.pdmodel，*.pdparams，*.pdopt
                # paddle.save(model, './save_model/train/train_{}.pdmodel'.format(epoch_id))
                print('./save_model/train/train_{}.pdparams'.format(epoch_id))
                paddle.save(model.state_dict(), './save_model/train/train_{}.pdparams'.format(epoch_id))
                paddle.save(opt_function.state_dict(), './save_model/train/train_{}.pdopt'.format(epoch_id))

                # 推理模型*.pdmodel和*.pdiparams
                # path = "./save_model/infer/infer"
                # paddle.jit.save(model, path,input_spec=[InputSpec(shape=[None, 3, INPUT_SIZE[0], INPUT_SIZE[1]], dtype='float32')])

            if epoch_id % 10 == 0:
                del_models()

    def eval(self, model_id):
        model = CatNn()
        model_state_dict = paddle.load('./save_model/train/train_{}.pdparams'.format(model_id))
        model.load_dict(model_state_dict)
        model.eval()  # 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False

        predata = ReadData()
        labels, imgs_path = predata.get_imgpath_label(label_text, train_dir)
        eval_dataset = LoadNnData(imgs_path=imgs_path, label_list=labels, hw_scale=INPUT_SIZE, model='eval')

        loader_eval = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
        os.system('rm -rf eval_result.csv')
        with open('eval_result.csv', 'w+', encoding='utf-8-sig', newline='') as csf:
            writer = csv.writer(csf)
            tp_num, in_num = 0., 0.
            for batch_id, data in enumerate(loader_eval()):
                lod_image, lod_labels, lod_imgs_path = data
                predicts = model(lod_image)
                for i, path in enumerate(lod_imgs_path):
                    name = path.split('/')[-1]
                    predict = np.argmax(predicts.numpy()[i])
                    gt = lod_labels.numpy()[i]
                    if gt == predict:
                        tp_num += 1
                    in_num += 1
                    writer.writerow([name, predict, gt, tp_num / in_num])
                    print('编号：{:>4}/2151 图片名称：{:<40} GT：{:<6}PT：{:<6} global accuracy:{:4.3f}'.format(
                        batch_id * BATCH_SIZE + i, name, gt, predict, tp_num / in_num))
            print('评估输入{}张图片，正确分类{}张图片，正确率：{:.3f}'.format(in_num, tp_num, tp_num / in_num))

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
            print('num_step, img_name, non,early,mid_advanced')
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


class ExportInfer(object):
    def __init__(self):
        """导出推理模型"""
        super(ExportInfer).__init__()

    def run(self, model_id):
        save_path = './save_model/infer/infer'  # 输出infer.pdmodel和infer.pdiparams
        load_path = './save_model/train/train_{}.pdparams'.format(model_id)
        model = CatNn()
        model_state_dict = paddle.load(load_path)
        model.load_dict(model_state_dict)
        model.eval()
        input_spec = [InputSpec(shape=[None, 3, INPUT_SIZE[0], INPUT_SIZE[1]], dtype='float32')]
        model = to_static(model, input_spec=input_spec)
        paddle.jit.save(model, save_path)


def run():
    # net = CatNn()
    # paddle.summary(net, (20, 3, INPUT_SIZE[0], INPUT_SIZE[1]))
    app().train()  # 由全局变量continue_id确定是否导入模型参数
    # app().eval(82)
    # app().predect(0)
    # ExportInfer().run(82)


if __name__ == '__main__':
    # 图像分类
    run()

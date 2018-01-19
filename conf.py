# -*- coding: utf-8 -*-
import os

# 输入图像位置数量
dim_image_L = 8
# 输入图像通道特征数量
#dim_image_D = 512
dim_image_D = 256

# 词特征向量维度
dim_embed = 512
# LSTM隐藏层维度
dim_hidden = 1024

batch_size = 32

n_epochs = 10000

# 测试时所采用的训练模型路径
test_model_path = './models/tensorflow_COCO/model-55'
# 测试时生成句子的最大长度
test_maxlen = 30
# 测试时若生成注意力高亮，大于多少的显示
CUT = 0.5

# 训练时模型的存储/加载路径
#model_path = './models/tensorflow_COCO'
model_path = './models/r152_tensorflow_COCO'

# vgg16 模型路径
vgg_path = './data/vgg16.tfmodel'
# 训练集路径
data_path = './data'

# 用vgg16 预先提取的图像特征路径
#feat_path = './data/COCO_feats.npy'
feat_path = './data/feats_resnet152.npy'

# 训练时模型的存储/加载路径
checkpoint_dir = model_path # './models/tensorflow_COCO'

annotation_path = os.path.join(data_path, 'caption_2.txt')
# log路径
log_path = './data/log/'

#测试的大小
TEST_IMAGE_SIZE = 2000

#COCO测试集图片的绝对路径
VAL_PATH = '/Users/lvxin/Downloads/Image_Captions/coco/val2014/'

#测试集的caption等
annFile = 'data/captions_val2014.json'

#存放测试时caption结果的文件
resFile = 'data/real_result.txt'

from model_coco import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 8"

train()

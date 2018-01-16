# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from test_COCO import *
import random
import json
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
from json import encoder
from conf import *
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

f = open('data/real_list.txt', 'r')
lines = f.readlines()
image_list = []
int_list = []
for line in lines:
	line = line.strip()
	image_list.append(line)
random.shuffle(image_list)
image_list = image_list[0 : TEST_IMAGE_SIZE]
# get image id
for i in range(len(image_list)):
	temp_str = image_list[i][13:25]
	#temp_str = image_list[i][15:27]
	temp_int = int(temp_str)
	int_list.append(temp_int)

json_res = []
for i in range(len(image_list)):
	image_dir = VAL_PATH + image_list[i]
	caption = captioning(image_dir)
	temp = {'image_id': int_list[i], 'caption': caption}
	json_res.append(temp)
	print 'Getting caption of image ' + str(i)
json_ans = json.dumps(json_res)
fw = open('data/real_result.txt', 'w')
fw.write(json_ans)
fw.close()


coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()
print '\n'
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)

import os

from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

from options.test_options import TestOptions
import sys
sys.path.insert(0, './image_segmentation')
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()
import sys
sys.path.insert(0, './image_synthesis')
import data
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

world_size = 1
rank = 0

# Corrects where dataset is in necesary format
opt.dataroot = '/kaggle/input/icnet-segmentation-output/results/temp'

opt.world_size = world_size
opt.gpu = 0
opt.mpdist = False

dataloader = data.create_dataloader(opt, world_size, rank)


model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt, rank)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1], gray=True)

webpage.save()

#synthesis_fdr = os.path.join(opt.results_dir, 'synthesis')

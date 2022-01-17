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

assert_and_infer_cfg(opt, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# Get segmentation Net
opt.dataset_cls = cityscapes
net = network.get_net(opt, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Segmentation Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
net.eval()
print('Segmentation Net Restored.')

# Get RGB Original Images
data_dir = opt.demo_folder
images = os.listdir(data_dir)
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

# Transform images to Tensor based on ImageNet Mean and STD
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

# Create save directory
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

soft_fdr = os.path.join(opt.results_dir, 'entropy')
soft_fdr_2 = os.path.join(opt.results_dir, 'logit_distance')
semantic_fdr = os.path.join(opt.results_dir, 'semantic')

if not os.path.exists(soft_fdr):
    os.makedirs(soft_fdr)
    
if not os.path.exists(soft_fdr_2):
    os.makedirs(soft_fdr_2)
    
if not os.path.exists(semantic_fdr):
    os.makedirs(semantic_fdr)
    
# creates temporary folder to adapt format to image synthesis
if not os.path.exists(os.path.join(opt.results_dir, 'temp')):
    os.makedirs(os.path.join(opt.results_dir, 'temp'))
    os.makedirs(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val'))
    os.makedirs(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val'))

softmax = torch.nn.Softmax(dim=1)

# Loop around all figures
for img_id, img_name in enumerate(images):
    img_dir = os.path.join(data_dir, img_name)
    img = Image.open(img_dir).convert('RGB')
    img.save(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val', img_name[:-4] + '_leftImg8bit.png'))
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(img_tensor.unsqueeze(0).cuda())
        print('%04d/%04d: Segmentation Inference done.' % (img_id + 1, len(images)))

        outputs = softmax(pred)
        
        # get entropy
        softmax_pred = torch.sum(-outputs*torch.log(outputs), dim=1)
        softmax_pred = (softmax_pred - softmax_pred.min()) / softmax_pred.max()
        
        # get logit distance
        distance, _ = torch.topk(outputs, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        map_logit = 1 - (result - result.min()) / result.max()

    pred_og = pred.cpu().numpy().squeeze()
    softmax_pred_og = softmax_pred.cpu().numpy().squeeze()
    map_logit = map_logit.cpu().numpy().squeeze()
    pred = np.argmax(pred_og, axis=0)

    softmax_pred_og = softmax_pred_og* 255
    map_logit = map_logit * 255
    pred_name = 'entropy_' + img_name
    pred_name_2 = 'distance_' + img_name
    pred_name_3 = 'pred_mask_' + img_name
    cv2.imwrite(os.path.join(soft_fdr, pred_name), softmax_pred_og)
    cv2.imwrite(os.path.join(soft_fdr_2, pred_name_2), map_logit)
    cv2.imwrite(os.path.join(semantic_fdr, pred_name_3), pred)
    
    # save label-based predictions, e.g. for submission purpose
    label_out = np.zeros_like(pred)
    for label_id, train_id in opt.dataset_cls.id_to_trainid.items():
        label_out[np.where(pred == train_id)] = label_id
    cv2.imwrite(os.path.join(semantic_label_fdr, pred_name), label_out)
    cv2.imwrite(os.path.join(semantic_fdr, pred_name), pred)
    cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_instanceIds.png'), label_out)
    cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_labelIds.png'), label_out)

print('Segmentation Results saved.')

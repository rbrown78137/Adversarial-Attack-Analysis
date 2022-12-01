from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# datasets.CocoDetection('./data/COCO/images', './data/COCO/annotations')
# model = SSD()

import torchattacks

model = ssdlite320_mobilenet_v3_large(pretrained=True)
model2 = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model2.eval()
x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
# If, images are normalized:
# atk.set_normalization_used(mean=[...], std=[...])
# adv_images = atk(images, labels)
predictions = model(x)
predictions2 = model2(x)
test = 1
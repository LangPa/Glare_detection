#!/usr/bin/env python

# Importing necessary packages:
import numpy as np
from time import time
import sys
from PIL import Image

try:
	import torch
	from torchvision import datasets, transforms, models

except:
    print("Install Pytorch: \nrun 'conda install pytorch torchvision cpuonly -c pytorch' on terminal \nFor more info go to: https://pytorch.org/")


# Define model and load model
model = models.resnet34(pretrained=True)
load = torch.load('model_glare.pt')
model.fc = torch.nn.Sequential(load['layers'])
model.fc.load_state_dict(load['state_dict'])

# get path to image directory:
try:
	image = Image.open(sys.argv[1])
except:
	img_dir = input(f'Could not find {sys.argv[1]}\nPlease give the image folder file path: ')

# Define transforms
transform  = transforms.Compose([     transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Run model through images
for image in sys.argv[1:]:
	img_path = img_dir + '/' + image
	try:
		image = Image.open(img_path)
	except:
		print('No image found at: ' + img_path)
		break

	# convert to tensor
	tensor = transform(image)
	tensor = tensor.unsqueeze(0)
	with torch.no_grad():
		ps = torch.exp(model(tensor))
		top_p, top_class = ps.topk(1, dim = 1)
		print(top_class.item())
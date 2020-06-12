#!/usr/bin/env python

# Importing necessary packages:
import numpy as np
from time import time
import sys
from PIL import Image
import os
import matplotlib.pyplot as plt

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
model.eval()

# Define transforms
transform  = transforms.Compose([     transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# Run model through images
t0 = time()
for image in sys.argv[1:]:
	
	img_path = None

	for path in ['flare', 'good']:
		if image in os.listdir(os.path.join('test', path)):
			img_path = os.path.join('test', path, image)

	if img_path:
		image = Image.open(img_path)
	else:
		print('No image found')
		break

	# convert to tensor
	tensor = transform(image)
	tensor = tensor.unsqueeze(0)

	with torch.no_grad():
		ps = torch.exp(model(tensor))
		top_p, top_class = ps.topk(1, dim = 1)
		print(top_class.item())

# print(f'time: {time() - t0}')
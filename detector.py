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

# get path to image directory:
# try:
# 	image = Image.open(sys.argv[1])
# except:
# 	img_dir = input(f'Could not find {sys.argv[1]}\nPlease give the image folder file path: ')

# Define transforms
transform  = transforms.Compose([     transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Run model through images
t0 = time()

batch = torch.tensor([])
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

	batch = torch.cat([batch, tensor])
	# plt.imshow(tensor.numpy().transpose((1, 2, 0)))
	# print(tensor)


	# print(tensor.shape)
	with torch.no_grad():
		ps = torch.exp(model(tensor))
		top_p, top_class = ps.topk(1, dim = 1)
		print(top_p.item(),top_class.item())


print('batch', batch.shape)
with torch.no_grad():
	ps = torch.exp(model(batch))
	top_p, top_class = ps.topk(1, dim = 1)
	for i in range(len(top_class)):
		print(top_class[i].item(), top_p[i].item())




# Loading in the data
data_dir = 'test'

# Convert to a torch tensor and normalise
transform  = transforms.Compose([     transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
dataset = datasets.ImageFolder(data_dir, transform=transform, target_transform = lambda x: (x+1) % 2)

testloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)

print('loader')
for image, labels in testloader:
	with torch.no_grad():
		ps = torch.exp(model(image))
		top_p, top_class = ps.topk(1, dim = 1)
		for i in range(len(top_class)):
			if labels[i].item() == 1:
				continue
			print(top_class[i].item(), top_p[i].item())
	# print(image.shape)
	# image = image.squeeze(0)
	# for image2 in batch:
	# 	print(image.shape)
	# 	if image2.numpy().all() == image.numpy().all():
	# 		print(True)

print(f'time: {time() - t0}')
i = input('s')
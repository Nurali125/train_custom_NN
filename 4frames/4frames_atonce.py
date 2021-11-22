from __future__ import print_function, division
from threading import Thread
import cv2
from cv2 import cuda
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import copy
import torch.nn.functional as F
from torchvision.transforms import functional


class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		self.stream = cv2.VideoCapture(src)
		# self.stream.set(3, 1080)  # 6x4 224p blocks = 24 blocks
		# self.stream.set(4, 720)
		(self.grabbed, self.frame) = self.stream.read()
		self.name = name
		self.stopped = False

	def start(self):
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True
		self.stream.release()



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'C:/Users/tileu/Desktop/Three_classes/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
labels = image_datasets['train'].classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft = torch.load('C:/Users/tileu/Desktop/Resnet18_224res_Last.pt')
model_ft.eval()

transform = transforms.Compose([            
 transforms.Resize((240)),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])


stream = WebcamVideoStream(src=0)
stream.start()
gpu = cv2.cuda_GpuMat()

while True:
	start = time.perf_counter()
	frame = stream.read()
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


	top_left = img[0:224, 0:224]
	top_right = img[0:224, 224:448] 
	bot_left = img[224:448, 0:224] 
	bot_right = img[224:448, 224:448]

	# top_left = cv2.cuda_GpuMat(img, (0,0,224,224))
	# top_left = cv2.UMat(img, [0, 224], [0, 224])
	# top_right = cv2.UMat(img, [0, 224], [224, 448])
	# bot_left = cv2.UMat(img, [224, 448], [0, 224])
	# bot_right = cv2.UMat(img, [224, 448], [224, 448])




	# top_left = top_left.download()
	# top_right = top_right.download()
	# bot_left = bot_left.download()
	# bot_right = bot_right.download()


	# arr_imgs = [top_left, bot_right]

	arr_imgs = [top_left, top_right, bot_left, bot_right]
	arr_lbl = ["", "", "", ""]
	prcnt = [0, 0, 0, 0]



	for i in range(4):
		im_pil = Image.fromarray(arr_imgs[i])
		img_t = transform(im_pil)
		batch_t = torch.unsqueeze(img_t, 0)
		model_ft.to(device)
		batch_t = batch_t.cuda()
		out = model_ft(batch_t)
		_, index = torch.max(out, 1)
		percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
		#     print(labels[index[0]], percentage[index[0]].item())
		lbl = labels[index[0]]
		#     prcnt = str(percentage[index[0]].item())
		prcnt = "{:.2f}".format(percentage[index[0]].item())
		Prediction = lbl + " " + prcnt
		_, indices = torch.sort(out, descending=True)
		#    print( [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]] )
		if i==0:
		    cv2.putText(frame, Prediction, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
		    cv2.rectangle(frame, (0,0), (224, 224), (0, 0, 255), 1)
		elif i==1:
		    cv2.putText(frame, Prediction, (240, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
		    cv2.rectangle(frame, (224, 0), (448, 224), (0, 0, 255), 1)
		elif i==2:
		    cv2.putText(frame, Prediction, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 
		    cv2.rectangle(frame, (0, 224), (224, 448), (0, 0, 255), 1)         
		elif i==3:
		    cv2.putText(frame, Prediction, (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))     
		    cv2.rectangle(frame, (224, 224), (448, 448), (0, 0, 255), 1)

	x = 1/(time.perf_counter() - start)
	cv2.putText(frame, str(int(x)) + " fps", (224, 224), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
	cv2.imshow("camera", frame)
	c = cv2.waitKey(1)
	if c == 27:
		break

stream.stop()
cv2.destroyAllWindows()

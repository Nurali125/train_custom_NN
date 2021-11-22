from __future__ import print_function, division
from threading import Thread
from multiprocessing import Lock, Process, Queue, current_process
import cv2
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


class WebcamVideoStream:
	def __init__(self, src=0, name = "C:/Users/tileu/Desktop/back1.mp4"):
		self.stream = cv2.VideoCapture(name)
		# self.stream.set(3, 1080)  # 6x4 224p blocks = 24 blocks
		# self.stream.set(4, 720)
		if (self.stream.isOpened()== False):
			print("Error opening video stream or file")

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
			if (self.stream.isOpened()== False):
				print("Error opening video stream or file")
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True
		self.stream.release()



def main():
	# Initialization
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

	# transform = transforms.Compose([            
	#  transforms.Resize((240)),                    
	#  transforms.CenterCrop(224),                
	#  transforms.ToTensor(),                     
	#  transforms.Normalize(                      
	#  mean=[0.485, 0.456, 0.406],                
	#  std=[0.229, 0.224, 0.225]                  
	#  )])

	transform = transforms.Compose([                           
	transforms.ToTensor(),                     
	transforms.Normalize(                      
	mean=[0.485, 0.456, 0.406],                
	std=[0.229, 0.224, 0.225]                  
	)])

	stream = cv2.VideoCapture(0)
	stream.set(3, 1080)
	stream.set(4, 720)

	# stream = WebcamVideoStream(name = "C:/Users/tileu/Desktop/back1.mp4")
	# stream.start()

	while True:
		start = time.perf_counter()
		rr, frame = stream.read()
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		arr_imgs = []
		for i in range(3):
			for j in range(4):
				arr_imgs.append(img[(i*224):((i+1)*224), (j*224):((j+1)*224)])


		# arr_imgs = (img[0:360, 0:480], img[0:360, 480:960], img[0:360, 960:1440], img[0:360, 1440:1920], 
		# 	img[360:720, 0:480], img[360:720, 480:960], img[360:720, 960:1440], img[360:720, 1440:1920],
		# 	img[720:1080, 0:480], img[720:1080, 480:960], img[720:1080, 960:1440], img[720:1080, 1440:1920])

		# arr_imgs = (img[0:224, 0:224], img[0:224, 224:448], img[0:224, 448:672], 
		# 	img[224:448, 0:224], img[224:448, 224:448], img[224:448, 448:672],
		# 	img[448:672, 0:224], img[448:672, 224:448], img[448:672, 448:672])


		crop_s = time.perf_counter() 
		crop_sec = time.perf_counter() - start

		for i in range(9):
			after_crop = time.perf_counter() 
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
				time_nn = time.perf_counter() - after_crop
				cv2.putText(frame, Prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
				cv2.rectangle(frame, (0,0), (224, 224), (0, 0, 255), 1)
				# cv2.putText(frame, str(time_nn) + "nn_sec", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
				# cv2.putText(frame, str(crop_sec) + "crop_sec", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
			elif i==1:
				cv2.putText(frame, Prediction, (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
				cv2.rectangle(frame, (224, 0), (448, 224), (0, 0, 255), 1)
			elif i==2:
				cv2.putText(frame, Prediction, (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255)) 
				cv2.rectangle(frame, (448, 0), (672, 224), (0, 0, 255), 1)    
			elif i==3:
				cv2.putText(frame, Prediction, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255)) 
				cv2.rectangle(frame, (0, 224), (224, 448), (0, 0, 255), 1)    
			elif i==4:
				cv2.putText(frame, Prediction, (240, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255))     
				cv2.rectangle(frame, (224, 224), (448, 448), (0, 0, 255), 1)
			elif i==5:
				cv2.putText(frame, Prediction, (460, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255)) 
				cv2.rectangle(frame, (448, 224), (672, 448), (0, 0, 255), 1)    

			elif i==6:
				cv2.putText(frame, Prediction, (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
				cv2.rectangle(frame, (0, 448), (224, 672), (0, 0, 255), 1)    
			elif i==7:
				cv2.putText(frame, Prediction, (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255))     
				cv2.rectangle(frame, (224, 448), (448, 672), (0, 0, 255), 1)
			elif i==8:
				cv2.putText(frame, Prediction, (460, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
				cv2.rectangle(frame, (448, 448), (672, 672), (0, 0, 255), 1)  

			# elif i==9:
			# 	cv2.putText(frame, Prediction, (460, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
			# 	cv2.rectangle(frame, (448, 448), (672, 672), (0, 0, 255), 1)
			# elif i==10:
			# 	cv2.putText(frame, Prediction, (460, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
			# 	cv2.rectangle(frame, (448, 448), (672, 672), (0, 0, 255), 1)
			# elif i==11:
			# 	cv2.putText(frame, Prediction, (460, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
			# 	cv2.rectangle(frame, (448, 448), (672, 672), (0, 0, 255), 1)


		x = 1/(time.perf_counter() - start)
		cv2.putText(frame, str(int(x)) + " fps", (224, 224), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
		cv2.imshow("camera", frame)
		c = cv2.waitKey(1)
		if c == 27:
			break

	stream.stop()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

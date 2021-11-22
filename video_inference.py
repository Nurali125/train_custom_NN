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
from queue import Queue




class NN_bro:
    def __init__(self, q_send, q_get, model_ft, labels, device):
        self.Prediction = ""
        self.model_ft = model_ft
        self.labels = labels
        self.device = device
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=(q_send, q_get, ))
        t.daemon = True
        t.start()
        return self

    def update(self, q_send, q_get):
        while True:
            if self.stopped:
                return

            frame = q_send.get()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)    
            # Neural Networks
            img_t = transform(im_pil)
            batch_t = torch.unsqueeze(img_t, 0)
            self.model_ft.to(self.device)
            batch_t = batch_t.cuda()
            out = self.model_ft(batch_t)
            _, index = torch.max(out, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            #     print(labels[index[0]], percentage[index[0]].item())
            lbl = self.labels[index[0]]
            #     prcnt = str(percentage[index[0]].item())
            prcnt = "{:.2f}".format(percentage[index[0]].item())
            self.Prediction = lbl + " " + prcnt
            q_get.put(self.Prediction)



    def read(self):
        return self.Prediction

    def frr(self):
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

	transform = transforms.Compose([            
	 transforms.Resize((240)),                    
	 transforms.CenterCrop(224),                
	 transforms.ToTensor(),                     
	 transforms.Normalize(                      
	 mean=[0.485, 0.456, 0.406],                
	 std=[0.229, 0.224, 0.225]                  
	 )])


	# stream = WebcamVideoStream(name = "C:/Users/tileu/Desktop/back1.mp4")
	# stream.start()

	q_send = Queue()
	q_get = Queue()

	cap = cv2.VideoCapture("C:/Users/tileu/Desktop/back1.mp4")
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	ret, frame = cap.read()
	q_send.put(frame)
	nn = NN_bro(q_send, q_get, model_ft, labels, device)
	nn.start()





	while True:
		start = time.perf_counter()
		ret, frame = cap.read()
		# frame = stream.read()
		
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		for i in range(3):
			for j in range(3):
				q_send.put(img[(i*224):((i+1)*224), (j*224):((j+1)*224)])


		for i in range(9):
			if not q_get.empty():
				Prediction = q_get.get()
				if i==0:
					cv2.putText(frame, Prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
					cv2.rectangle(frame, (0,0), (224, 224), (0, 0, 255), 1)
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

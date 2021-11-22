from __future__ import print_function, division
from threading import Thread
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
    def __init__(self, q, model_ft, labels, device):
        self.Prediction = ""
        self.model_ft = model_ft
        self.labels = labels
        self.device = device
        self.stopped = False
        self.arr_pred = []
        self.got_frame = None

    def start(self):
        t = Thread(target=self.update, args=(q, ))
        t.daemon = True
        t.start()
        return self

    def update(self, q):
        while True:
            if self.stopped:
                return
            
            if (len(self.arr_pred) >= 9):
                self.arr_pred = []
            frame = q.get()
            q.task_done()
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

    def read(self):
        return self.Prediction

    def stop(self):
        self.stopped = True




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


q = Queue()
q_other = Queue()

stream = cv2.VideoCapture(0)
stream.set(3, 1080)
stream.set(4, 720)
ff = 0

Prediction =[]
while True:
    start = time.perf_counter()
    red, frame = stream.read()

    if ff==0:
        q.put(frame)
    else:
        for i in range(3):
            for j in range(3):
                xy = frame[(i*224):((i+1)*224), (j*224):((j+1)*224)]
                q.put(xy)

    if ff==0:
        nn = NN_bro(q, model_ft, labels, device)
        nn.start()
    ff = 7

    for p in range(9):
        previous = Prediction[len(Prediction)-1]
        if (len(Prediction) >= 9):
            Prediction =[]
        current = nn.read()
        if()

        Prediction.append()
        if p==0:
            aa = str(Prediction[0])
            # time_nn = time.perf_counter() - after_crop
            cv2.putText(frame, aa, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
            cv2.rectangle(frame, (0,0), (224, 224), (0, 0, 255), 1)
            # cv2.putText(frame, str(time_nn) + "nn_sec", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
            # cv2.putText(frame, str(crop_sec) + "crop_sec", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
        elif p==1:
            aa = str(Prediction[1])
            cv2.putText(frame, aa, (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255))
            cv2.rectangle(frame, (224, 0), (448, 224), (0, 0, 255), 1)
        elif p==2:
            aa = str(Prediction[2])
            cv2.putText(frame, aa, (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255)) 
            cv2.rectangle(frame, (448, 0), (672, 224), (0, 0, 255), 1)    
        elif p==3:
            aa = str(Prediction[3])
            cv2.putText(frame, aa, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255)) 
            cv2.rectangle(frame, (0, 224), (224, 448), (0, 0, 255), 1)    
        elif p==4:
            aa = str(Prediction[4])
            cv2.putText(frame, aa, (240, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255))     
            cv2.rectangle(frame, (224, 224), (448, 448), (0, 0, 255), 1)
        elif p==5:
            aa = str(Prediction[5])
            cv2.putText(frame, aa, (460, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255)) 
            cv2.rectangle(frame, (448, 224), (672, 448), (0, 0, 255), 1)    
        elif p==6:
            aa = str(Prediction[6])
            cv2.putText(frame, aa, (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
            cv2.rectangle(frame, (0, 448), (224, 672), (0, 0, 255), 1)    
        elif p==7:
            aa = str(Prediction[7])
            cv2.putText(frame, aa, (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255))     
            cv2.rectangle(frame, (224, 448), (448, 672), (0, 0, 255), 1)
        elif p==8:
            aa = str(Prediction[8])
            cv2.putText(frame, aa, (460, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255)) 
            cv2.rectangle(frame, (448, 448), (672, 672), (0, 0, 255), 1) 
    

    x = 1/(time.perf_counter() - start)
    cv2.putText(frame, str(int(x)) + " fps", (224, 224), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imshow("camera", frame)
    
    c = cv2.waitKey(1)
    if c == 27:
        break

stream.release()
cv2.destroyAllWindows()

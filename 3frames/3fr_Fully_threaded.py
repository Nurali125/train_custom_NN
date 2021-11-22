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

class WebcamVideoStream:
    def __init__(self, q):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(3, 1920)  # 6x4 224p blocks = 24 blocks
        self.stream.set(4, 1080)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=(q, ))
        t.daemon = True
        t.start()
        return self

    def update(self, q):
        while True:
            st = time.perf_counter()
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            x = ("full", self.frame)
            q.put(x)

            frame1 = self.frame[0:224, 0:224]
            y = ("1st", frame1)
            q.put(y)

            # frame2 = self.frame[0:224, 224:448]
            # z = ("2nd", frame2)
            # q.put(z)

            # frame3 = self.frame[0:224, 448:672]
            # w = ("3d", frame3)
            # q.put(w)

            xxx = (time.perf_counter() - st)
            print(xxx, "fps Webcam")


    def read(self):
        return self.frame


    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()



class NN_bro:
    def __init__(self, q, q_get, model_ft, labels, device):
        self.Prediction = ""
        self.model_ft = model_ft
        self.labels = labels
        self.device = device
        self.stopped = False

        self.got_frame = None

    def start(self):
        t = Thread(target=self.update, args=(q, q_get, ))
        t.daemon = True
        t.start()
        return self

    def update(self, q, q_get):
        while True:
            st = time.perf_counter()
            if self.stopped:
                return

            frame = q.get()
            img = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
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
            if (frame[0] == "full"):
                zz = (frame[0], self.Prediction, frame[1])
                q_get.put(zz)
            else:
                zz = (frame[0], self.Prediction)
                q_get.put(zz)

            xxx = (time.perf_counter() - st)
            print(xxx, "fps NN inference")


    def read(self):
        return self.Prediction

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


q = Queue()
q_get = Queue()

stream = WebcamVideoStream(q)
stream.start()

nn = NN_bro(q, q_get, model_ft, labels, device)
nn.start()

kkk = 0
pred_arr = []
while True:
    start = time.perf_counter()
    print(kkk)
    prediction = q_get.get()
    pred_arr.append(prediction)
    if(len(pred_arr) == 2):
        # frame = stream.read()
        frame = pred_arr[0][2]
        # pred_arr[3][0] --- 3 -> 3rd element of array; 0th element of tuple "prediction" -> string defining which frame (number)
        # pred_arr[3][1] --- 3 -> 3rd element of array; 1st element of tuple "prediction" -> string defining percentage and prediction class

        
        if(pred_arr[1][0] == "1st"):
            # print(pred_arr[1][0], pred_arr[1][1], "iteration: ", kkk)
            cv2.putText(frame, pred_arr[1][1], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.rectangle(frame, (0,0), (224, 224), (0, 0, 255), 1)


        # if(pred_arr[2][0] == "2nd"):
        #     # print(pred_arr[2][0], pred_arr[2][1], "iteration: ", kkk)
        #     cv2.putText(frame, pred_arr[2][1], (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        #     cv2.rectangle(frame, (224, 0), (448, 224), (0, 0, 255), 1)

        # if(pred_arr[3][0] == "3rd"):
        #     # print(pred_arr[3][0], pred_arr[3][1], "iteration: ", kkk)
        #     cv2.putText(frame, pred_arr[3][1], (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        #     cv2.rectangle(frame, (448, 0), (672, 224), (0, 0, 255), 1)

        if(pred_arr[0][0] == "full"):
            # print(pred_arr[0][0], pred_arr[0][1], "iteration: ", kkk)
            cv2.imshow("frame1", frame)

        pred_arr = []  # EMPTY BUFFER OF PREDICTIONS
        
        c = cv2.waitKey(1)
        if c == 27:
            break
        

    # Prediction1 = q_get.get()
    # cv2.putText(frame1, Prediction1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # prediction = q_get.get()
    # print(prediction[0], prediction[1], "iteration: ", kkk)


    # prediction = q_get.get()
    # print(prediction[0], prediction[1], "iteration: ", kkk)


    # prediction = q_get.get()
    # print(prediction[0], prediction[1], "iteration: ", kkk)
    # Prediction2 = q_get.get()
    # if (Prediction1 == Prediction2):
    #     Prediction2 = None
    # else:
    #     cv2.putText(frame2, Prediction2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    
    # cv2.putText(frame3, Prediction3, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    x = (time.perf_counter() - start)
    print(x, "fps main func")
    # cv2.putText(frame2, str(int(x)) + " fps", (224, 224), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    # cv2.imshow("frame1", frame1)
    # cv2.imshow("frame2", frame2)
    # cv2.imshow("frame3", frame3)
    
    kkk += 1


stream.release()
cv2.destroyAllWindows()

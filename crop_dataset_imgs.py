import os, cv2
import random

# directory = "C:/Users/tileu/Desktop/Three_classes/train/people"
directory = "C:/Users/tileu/Desktop/images/buildings"


for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff"):
        # print(filename)
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        height = img.shape[0]
        width = img.shape[1]

        print(height, width)

        if (height<180 and width<180):
            # n = random.randint(0,9999)
            # cv2.imwrite("C:/Users/tileu/Desktop/images/LessThan180/"+str(height)+str(n)+".jpg", img)
            print("height and width < 180")
        elif (height<180 and 180<=width<350):
            # n = random.randint(0,9999)
            # cv2.imwrite("C:/Users/tileu/Desktop/images/LessThan180/"+str(n)+str(width)+".jpg", img)
            print("height<180, but width>180")
        elif (height<180 and width>=350):
            # n = random.randint(0,9999)
            # cv2.imwrite("C:/Users/tileu/Desktop/images/LessThan180/"+str(n)+str(width)+".jpg", img)
            print("height<180, but width>180")


        elif (180<=height<350 and width<180):
            # n = random.randint(0,9999)
            # cv2.imwrite("C:/Users/tileu/Desktop/images/LessThan180/"+str(n)+str(width)+".jpg", img)
            print("height>180, but width<180")
        elif (180<=height<350 and 180<=width<350):
            # n = random.randint(0,9999)
            # cv2.imwrite("C:/Users/tileu/Desktop/images/Normal/"+str(width), img)
            print("height and width BOTH between 180 and 350")
        elif (180<=height<350 and width>=350):
            w = round(width/224)
            ww = int(width/w)
            print("180<=height<350 and width>=350")
            for i in range(w):
                image = img[0:height, i*ww:(i+1)*ww]
                cv2.imwrite("C:/Users/tileu/Desktop/images/Cropped/"+str(i)+str(height)+".jpg", image)


        elif (height>=350 and width<180):
            print("height>=350 and width<180")
            h = round(height/224)
            hh = int(height/h)
            for j in range(h):
                image = img[j*hh:(j+1)*hh, 0:width]
                cv2.imwrite("C:/Users/tileu/Desktop/images/Cropped/"+str(h)+str(hh)+str(j)+str(width)+".jpg", image)
        elif (height>=350 and 180<=width<350):
            h = round(height/224)
            hh = int(height/h)
            for j in range(h):
                image = img[j*hh:(j+1)*hh, 0:width]
                cv2.imwrite("C:/Users/tileu/Desktop/images/Cropped/"+str(h)+str(hh)+str(j)+str(width)+".jpg", image)
            print("height>=350 and 180<=width<350")
        elif (height>=350 and width>=350):
            n = random.randint(0,9999)
            w = round(width/224)
            h = round(height/224)
            ww = int(width/w)
            hh = int(height/h)
            print("height>=350 and width>=350")
            for j in range(h):
                for i in range(w):
                    image = img[(j*hh):((j+1)*hh), (i*ww):((i+1)*ww)]
                    # print(j*hh, ((j+1)*hh), (i*ww), ((i+1)*ww))
                    # print("i,j:", i, j)
                    cv2.imwrite("C:/Users/tileu/Desktop/images/Cropped/" + str(i)+str(j)+str(n) + ".jpg", image)
                    

        else:
            print("else case")


    else:
        continue

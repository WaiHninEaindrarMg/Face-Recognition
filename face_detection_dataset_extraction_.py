
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt


main_path = r'.\video_test'

dataset_folder = r'.\dataset1'
train_folder = os.path.join(dataset_folder,'train')
val_folder = os.path.join(dataset_folder,'validation')

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
    os.makedirs(train_folder)
    os.makedirs(val_folder)
    

model = YOLO("yolov8n-face.pt")


def face_detect(img):
    detection = model(img)
    boxes = detection[0].boxes

    return boxes

def draw_box(boxes,img):

    for box in boxes:
        x = int(box.xyxy.tolist()[0][0])
        y = int(box.xyxy.tolist()[0][1])
        w = int(box.xyxy.tolist()[0][2])
        h = int(box.xyxy.tolist()[0][3])
           
        cv2.rectangle(img,(x,y),(w,h),(50,200,129),2)
    return img






for file in os.listdir(main_path):
    if file.endswith('.mp4'):
        name = file.split(".")[0]
        video_file = os.path.join(main_path,file)

        train_name_folder = os.path.join(train_folder,name)
        val_name_folder = os.path.join(val_folder,name)
        if not os.path.exists(train_name_folder):
            os.makedirs(train_name_folder)
        if not os.path.exists(val_name_folder):
            os.makedirs(val_name_folder)

        print(video_file)
        count = 0
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print("error opening video file")
            exit()
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"FPS : {fps}, width:{frame_width} , height : {frame_height}")

        while True:
            ret , frame = cap.read()

            if not ret:
                break

            frame = np.array(frame,"uint8")
            boxes = face_detect(frame)
            for box in boxes:
                x = int(box.xyxy.tolist()[0][0])
                y = int(box.xyxy.tolist()[0][1])
                w = int(box.xyxy.tolist()[0][2])
                h = int(box.xyxy.tolist()[0][3])
                box_area = (w-x) * (h-y)
                print("Bounding Box Area:", box_area)
            if box_area > 30000:
                count = count + 1
                frame_ = draw_box(boxes,frame)
                cv2.imshow(f"Video {name}",frame_)
                img = frame[y:h,x:w]
                crop_img = cv2.resize(img,(224,224))
                
                if count <= 800:
                    img_name = os.path.join(train_name_folder,f"{name}_{count}.jpg")
                    cv2.imwrite(img_name,crop_img)
                elif count > 800 and count <=900:
                    img_name = os.path.join(val_name_folder,f"{name}_{count}.jpg")
                    cv2.imwrite(img_name,crop_img)


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
cap.release()
cv2.destoryAllWindows()

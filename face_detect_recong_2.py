import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model



model = YOLO("yolov8n-face.pt")
res = "152"
loaded_model = load_model(f"six_face_resnet{res}.h5")


def face_detect(img):
    detection = model(img)
    boxes = detection[0].boxes

    return boxes

def draw_box(boxes,img,thick=2):

    for box in boxes:
        x = int(box.xyxy.tolist()[0][0])
        y = int(box.xyxy.tolist()[0][1])
        w = int(box.xyxy.tolist()[0][2])
        h = int(box.xyxy.tolist()[0][3])
           
        cv2.rectangle(img,(x,y),(w,h),(50,200,129),thick)
    return img

def draw_box_label(boxes,img,label,font_scale=2,font_thickness=3,boxthick=2):

    for box in boxes:
        x = int(box.xyxy.tolist()[0][0])
        y = int(box.xyxy.tolist()[0][1])
        w = int(box.xyxy.tolist()[0][2])
        h = int(box.xyxy.tolist()[0][3])     

        cv2.rectangle(img,(x,y),(w,h),(50,200,129),boxthick)


        # Add label text
        label = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x + 5  # Adjust the x-coordinate to position the text to the right of the box
        text_y = y - 10  # Adjust the y-coordinate to position the text above the box
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (50, 200, 129), font_thickness)


    return img



files = r'.\video_test'
# Define the codec and create VideoWriter object
width , height = 640, 380
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs as well
out = cv2.VideoWriter(f'.\output_video{res}.avi', fourcc, 15.0, (width, height))  # Adjust frame size and frame rate as needed

for file in os.listdir(files):
    if file.endswith('.mp4'):
        name = file.split(".")[0]
        video_file = os.path.join(files,file)
        print(video_file)
        count = 0
        
        cap = cv2.VideoCapture(video_file)

        f_cnt = cv2.CAP_PROP_FRAME_COUNT
        numFrames = int(cap.get(f_cnt))
        print(f"Numframes : {numFrames}")

        skip_frames = 15
        frame_count = 0


        if not cap.isOpened():
            print("error opening video file")
            exit()
        

        for i in range(0,numFrames, skip_frames):
        
            ret , frame = cap.read()

            if not ret:
                break

            frame_count += 1
            
        
            frame = np.array(frame,"uint8")
            #frame = cv2.resize(frame,(width, height))
            boxes = face_detect(frame)

            for box in boxes:
                x = int(box.xyxy.tolist()[0][0])
                y = int(box.xyxy.tolist()[0][1])
                w = int(box.xyxy.tolist()[0][2])
                h = int(box.xyxy.tolist()[0][3])
                box_area = (w-x) * (h-y)
                print("Bounding Box Area:", box_area)

            if box_area > 30:
            
                count = count + 1
                
                img = frame[y:h,x:w]
                crop_img = cv2.resize(img,(224,224))
                
                img_array = image.img_to_array(crop_img)
                img_array = np.expand_dims(img_array, axis = 0)
                img_array = img_array / 255.0

                # prediction 
                predictions = loaded_model.predict(img_array)
                predicted_class = np.argmax(predictions)
                # map predict class
                class_labels = ['cillian murphy','Elon','Rashford','Robert','Ronaldo','Zlatan']
                predicted_label = class_labels[predicted_class]
                
                frame_ = draw_box_label(boxes,frame,predicted_label,font_scale=5,font_thickness=5,boxthick=5)
                frame_ = cv2.resize(frame_,(width, height))
                out.write(frame_)
                cv2.imshow(f"Video {name}",frame_)
                    

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
cap.release()
out.release()
cv2.destoryAllWindows()

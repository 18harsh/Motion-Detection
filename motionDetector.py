import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas
import seaborn
import time
from datetime import datetime


first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])
# model = keras.models.load_model('saved_models/MY-cifar10-trained-model.h5')
model = keras.models.load_model('../project02/saved_models/keras_fashionshape(28,28,1)-trained-model.h5') 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# category = ["Airplanes"," Cars" ,"Birds" ,"Cats"," Deer", "Dogs", "Frogs", "Horses ","Ships"," Trucks"]


video=cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 150, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1

        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    # cv2.imshow("Gray Frame",gray)
    # cv2.imshow("Delta Frame",delta_frame)
    # cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    #Machine Learning

    # img_array = cv2.resize(frame,(32,32))
    # img_array = img_array.astype('float32')
    # img_array = img_array/255
    # img_array = img_array.reshape(-1,32,32,3)
    # val = model.predict_classes(img_array)
    # print(category[np.array(val)[0]])
    
    def get_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(img,(28,28))

    try:
        # crop_img = gray[y:y+h, x:x+w]
        images = np.array([get_image(frame)])
        images_reshaped = images.reshape(images.shape[0], 28, 28, 1)
        images_reshaped = tf.cast(images_reshaped, tf.float32)
        preds = model.predict(images_reshaped)
        if 100*np.max(preds)>=80:
            predicted_label = np.argmax(preds[0])
            print(class_names[predicted_label])
    except Exception as e:
        print(e)



    #Quit
    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break


print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows
plt.imshow(images[0])
# plt.imshow(crop_img)
plt.show()

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
model = tf.keras.models.load_model(r"GradioApp\MobileNet.h5")
labels = ['cam', 'chi_dan', 'hieu_lenh', 'nguy_hiem', 'phu']

def predict(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    h, w, _ = img.shape
    new_img = img.copy()
    new_img = np.expand_dims(new_img, axis = 0)
    (boxPreds, labelPred) = model.predict(new_img)
    (startX, startY, endX, endY) = boxPreds[0]
    startX = int(round((startX * w), 0))
    startY = int(round((startY * h), 0))
    endX = int(round((endX * w), 0))
    endY = int(round((endY * h), 0))
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return [{labels[i]: labelPred[0][i] for i in range(5)}, img]

#building the web application

iface = gr.Interface(
    fn = predict,
    inputs = gr.Image(type = "filepath"),
    outputs = [gr.Label(num_top_classes = 5), "image"]
)

#launching the web application
iface.launch()
# from enum import auto
import tkinter as tk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import time
import cv2
import cv2
import tkinter
import cv2
import PIL.Image
import PIL.ImageTk
import time
import pyautogui

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model("mask_detector.model")

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        
        self.vid = MyVideoCapture(self.video_source)

       
        self.canvas = tkinter.Canvas(
            window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

       
        self.btn_snapshot = tkinter.Button(
            window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_quit = tkinter.Button(
            window, text="Quit", width=50, command=self.quit)
        self.btn_quit.pack(anchor=tkinter.CENTER, expand=True)


        self.delay = 1
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        # ::::::::::::::::::::::::::::::
        # myScreenshot = pyautogui.screenshot()
        # myScreenshot.save(r'C:\Users\YASH\Desktop\Project\DesktopGUI\Tkinter\FaceMaskDetection Project\screenshot_1.png')
        # ::::::::::::::::::::::::::::::::
            x, y = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            pyautogui.screenshot('screenshot.png', region=(x+5, y+5, w-10, h-10))


        # ret, frame = self.vid.get_frame()

        # if ret:
        #     cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def quit(self):
        self.window.destroy()

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
       
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()
        # print(detections.shape)

        
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # print(confidence)
           
            if confidence > 0.5:
                # print(confidence)
               
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
    
               
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
               
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
    
               
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
           
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            # self.vd=  detect_and_predict_mask(frame, faceNet, maskNet)
            # frame = cv2.resize(frame, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
            (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
            
            for (box, pred) in zip(locs, preds):
               
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)

               
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), color, 2)
                # cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF
            
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        # else:
        #     return (ret, None)

   
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()



App(tkinter.Tk(), "Face Mask Detector")

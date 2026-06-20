import cv2
import urllib
import numpy as np
import os

# Load configuration
try:
    from config import CAMERA_URL, USE_WEBCAM
except ImportError:
    CAMERA_URL = "http://192.168.1.100:8080/shot.jpg"  # Default IP webcam URL
    USE_WEBCAM = False  # Set to True to use built-in webcam instead

classifier = cv2.CascadeClassifier(os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml"))

url = CAMERA_URL


data = []

while len(data) < 100:
    
    if USE_WEBCAM:
        # Use built-in webcam
        if 'cap' not in locals():
            cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
    else:
        # Use IP webcam
        try:
            image_from_url = urllib.request.urlopen(url)
            frame = np.array(bytearray(image_from_url.read()),np.uint8)
            frame = cv2.imdecode(frame,-1)
        except:
            print("Failed to grab frame from IP camera. Check URL in config.py")
            break
    
    face_points = classifier.detectMultiScale(frame,1.3,5)
    
    if len(face_points)>0:
        for x,y,w,h in face_points:
            face_frame = frame[y:y+h+1,x:x+w+1]
            cv2.imshow("Only face",face_frame)
            if len(data)<=100:
                print(len(data)+1,"/100")
                data.append(face_frame)
                break
    cv2.putText(frame, str(len(data)),(100,100),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255))
    cv2.imshow("frame",frame)
    if cv2.waitKey(30) == ord("q"):
        break
cv2.destroyAllWindows()
if USE_WEBCAM and 'cap' in locals():
    cap.release()
        
if len(data)== 100:
    name = input("Enter Face holder name : ")
    img_dir = os.path.join(os.getcwd(), 'images')
    os.makedirs(img_dir, exist_ok=True)
    for i in range(100):
        cv2.imwrite(os.path.join(img_dir, f"{name}_{i}.jpg"), data[i])
    print("Done")
else:
    print("need more data")
        
    


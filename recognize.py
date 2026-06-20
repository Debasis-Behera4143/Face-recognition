
import urllib
import cv2
import numpy as np
import os
import pickle
from keras.models import load_model

# Load configuration
try:
    from config import CAMERA_URL, USE_WEBCAM
except ImportError:
    CAMERA_URL = "http://192.168.1.100:8080/shot.jpg"  # Default IP webcam URL
    USE_WEBCAM = False  # Set to True to use built-in webcam instead

classifier = cv2.CascadeClassifier(os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml"))

model = load_model(os.path.join(os.getcwd(), "final_model.h5"))

URL = CAMERA_URL

# Load labels from training data
try:
    from sklearn.preprocessing import LabelEncoder
    with open(os.path.join(os.getcwd(), 'data', 'labels.p'), 'rb') as f:
        original_labels = pickle.load(f)
    le = LabelEncoder()
    le.fit(original_labels)
    LABEL_NAMES = list(le.classes_)
except:
    # Fallback if labels.p doesn't exist
    LABEL_NAMES = ["Person_0", "Person_1", "Person_2"]
    print("Warning: Could not load labels from data/labels.p. Using default labels.")

def get_pred_label(pred):
    return LABEL_NAMES[pred]

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255
    return img
    


if USE_WEBCAM:
    cap = cv2.VideoCapture(0)

ret = True
while ret:
    
    if USE_WEBCAM:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
    else:
        try:
            img_url = urllib.request.urlopen(URL)
            image = np.array(bytearray(img_url.read()),np.uint8)
            frame = cv2.imdecode(image,-1)
        except:
            print("Failed to grab frame from IP camera. Check URL in config.py")
            break
    
    faces = classifier.detectMultiScale(frame,1.5,5)
      
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(frame,get_pred_label(np.argmax(model.predict(preprocess(face)))),
                    (200,200),cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),2)
        
    cv2.imshow("capture",frame)
    if cv2.waitKey(1)==ord('q'):
        break

if USE_WEBCAM:
    cap.release()
cv2.destroyAllWindows()


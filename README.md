# Face Detection and Recognition System

A real-time face detection and recognition system using OpenCV and deep learning (CNN model based on LeNet architecture).

## Features

- Collect face data from webcam or IP camera
- Train a CNN model to recognize faces
- Real-time face recognition with live video feed
- Support for both built-in webcam and IP webcam (DroidCam, IP Webcam app, etc.)

## Prerequisites

Install the required Python packages:

```bash
pip install opencv-python numpy scikit-learn tensorflow keras matplotlib pillow
```

## Project Structure

```
Face_Detectiondev/
├── collect_data.py           # Collect face images for training
├── consolidated_data.py      # Process and consolidate image data
├── face_detection.py         # Train the face recognition model
├── recognize.py              # Real-time face recognition
├── config.py                 # Configuration file (camera settings)
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── data/                     # Folder for processed data (images.p, labels.p)
├── images/                   # Folder for collected face images
└── final_model.h5           # Trained model (generated after training)
```

## How to Use

### Step 1: Configure Camera Settings

Edit `config.py`:

- **For built-in webcam**: Set `USE_WEBCAM = True`
- **For IP webcam**: Set `USE_WEBCAM = False` and update `CAMERA_URL` with your IP webcam URL
  - Example: Install "IP Webcam" app on Android, start the server, and use the URL shown

### Step 2: Collect Face Data

Run the data collection script to capture 100 images per person:

```bash
python collect_data.py
```

- Position your face in front of the camera
- The script will automatically detect and capture 100 face images
- Press 'q' to quit early if needed
- Enter the person's name when prompted
- Repeat this process for each person you want to recognize

### Step 3: Consolidate Data

Process all collected images into training data:

```bash
python consolidated_data.py
```

This creates two files in the `data/` folder:
- `images.p` - Preprocessed face images
- `labels.p` - Corresponding labels/names

### Step 4: Train the Model

Train the face recognition model:

```bash
python face_detection.py
```

This will:
- Load the processed data
- Train a CNN model (10 epochs by default)
- Save the trained model as `final_model.h5`

### Step 5: Run Face Recognition

Run the real-time face recognition:

```bash
python recognize.py
```

- The system will detect and recognize faces in real-time
- Press 'q' to quit

## Camera Configuration

### Using Built-in Webcam
Set in `config.py`:
```python
USE_WEBCAM = True
```

### Using IP Webcam (Phone Camera)

1. Install an IP webcam app on your phone:
   - Android: "IP Webcam" by Pavel Khlebovich
   - iOS: "EpocCam" or similar apps

2. Connect your phone and computer to the same WiFi network

3. Start the server in the app and note the IP address (e.g., `http://192.168.1.100:8080`)

4. Update `config.py`:
```python
USE_WEBCAM = False
CAMERA_URL = "http://192.168.1.100:8080/shot.jpg"
```

## Model Architecture

The system uses a LeNet-inspired CNN architecture:
- Conv2D (30 filters, 5x5) + ReLU + MaxPooling
- Conv2D (15 filters, 3x3) + ReLU + MaxPooling
- Flatten
- Dense (50 units) + ReLU
- Dense (output layer) + Softmax

## Troubleshooting

**Issue**: "Failed to grab frame from IP camera"
- Check if your phone and computer are on the same network
- Verify the IP address in `config.py` matches the one shown in your IP webcam app
- Make sure the IP webcam server is running

**Issue**: "Failed to grab frame from webcam"
- Check if another application is using the webcam
- Try closing other apps that might be using the camera

**Issue**: Model accuracy is low
- Collect more diverse training data (different angles, lighting conditions)
- Increase training epochs in `face_detection.py`
- Ensure good lighting when collecting data

## Notes

- Collect face images in good lighting conditions for better accuracy
- Try to capture faces from different angles during data collection
- The model works best when trained with at least 2-3 different people
- For best results, use similar lighting conditions during recognition as during training

## License

This is an educational project. Feel free to modify and use for learning purposes.

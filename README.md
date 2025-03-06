# Face Mesh Landmark Detection in Video

## Description
This Python script detects and visualizes multiple faces and their respective facial landmarks from a video file using Google's MediaPipe library and OpenCV.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)

## Installation
```bash
pip install opencv-python mediapipe
```

## Usage
1. Place your input video file (e.g., `video.mp4`) in the same directory as the script.
2. Execute the script using:
```bash
python your_script_name.py
```

Press the `q` key to exit the video display window.

## Functionality
- Reads frames from a video file.
- Converts each frame from BGR to RGB format.
- Detects and visualizes facial landmarks for multiple faces (up to 5 per frame, adjustable).
- Displays real-time visualization of facial landmarks.

## Notes
- Adjust `max_num_faces` based on your video content needs.
- Ensure the video file (`video.mp4`) exists in the script's directory.

## Future Enhancements
- Add landmark connection visualization for improved facial mesh representation.
- Integrate error handling for frames without detected faces.
- Optimize performance for real-time webcam processing.

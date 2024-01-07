# drishtii

This Python script is designed for video processing using the YOLO (You Only Look Once) object detection model and a pre-trained deep learning model for activity recognition. The code utilizes the multiprocessing module to concurrently perform video capture, object detection, and pose-based activity recognition.

Key Components:

Video Capture (capture function):

Uses OpenCV to read frames from a specified video file (main__.mp4).
Initializes an initial frame for reference and sets up a shared queue (frame_queue) for communication between processes.
Utilizes the YOLO model from the Ultralytics library for real-time object detection.
Displays the annotated video feed with bounding boxes around detected objects and labels indicating tracking IDs and activities.
Allows the user to terminate the video feed by pressing 'q'.
Activity Recognition (poseD function):

Takes frames from the shared queue (frame_queue) produced by the capture function.
Resizes and normalizes each frame before appending it to a deque with a specified sequence length (SEQUENCE_LENGTH).
Applies a pre-trained deep learning model (model_new) to predict the activity (e.g., walking, stair climbing) based on the sequence of frames.
Updates a shared dictionary (poseD_) with the recognized activity, which is displayed in the video feed.
Main Execution:

Sets up multiprocessing with shared data structures (queues and dictionaries).
Spawns separate processes for video capture (capture function) and activity recognition (poseD function).
Enables concurrent execution of both processes.
Waits for processes to finish execution.
Overall, the script integrates real-time object detection with YOLO and activity recognition using a deep learning model, making it suitable for applications such as surveillance or monitoring activities in video streams. Note that the code may require proper configuration of model paths, and the Ultralytics and TensorFlow libraries need to be installed for the script to run successfully.






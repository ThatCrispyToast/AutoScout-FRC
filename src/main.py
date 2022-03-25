#!/usr/bin/env python3
"""Main Processing File"""

__author__ = "Aniket Chadalavada"
__version__ = "0.0.1"
__license__ = "MIT"

# Dependencies
import cv2
from tensorflow import lite
# Local Imports
from constants.req import *
from util import *

# Example Detection: [{'bounding_box': array([0.26929605, 0.5280721 , 0.48070395, 0.71024466], dtype=float32), 'class_id': 0.0, 'score': 0.796875}, {'bounding_box': array([0.27319047, 0.3893147 , 0.48459837, 0.55939335], dtype=float32), 'class_id': 1.0, 'score': 0.7578125}]

# Load Labels into List
classes = LABELS.copy()

def run_odt(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # Load the input image and preprocess it
    preprocessed_image = preprocess_image(
        image_path, (input_height, input_width)
    )[0] # Only Gets Preprocessed Image Instead of Both Processed and Original

    # Run Object Detection on Image
    return detect_objects(interpreter, preprocessed_image, threshold=threshold)


def main():
  # Load TFLite model
  interpreter = lite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
  interpreter.allocate_tensors()

  # Start Video Capture from Video
  vid = cv2.VideoCapture(DEFAULT_VID_PATH)

  while True:
      # Read Frame from Video
      ret, frame = vid.read()
      # Empty Frame Handling
      if not ret:
        print("Frame Skipped")
        continue
      # Write Frame to File for TF Processing
      cv2.imwrite("frame.jpg", frame)

      # Run inference and draw detection result on the local copy of the original file
      detection_result = run_odt("frame.jpg", interpreter, threshold=DETECTION_THRESHOLD)

      print(detection_result)

      # Exit on "q" Pressed
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break

  # Release Video and Destroy Lingering Frame
  vid.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
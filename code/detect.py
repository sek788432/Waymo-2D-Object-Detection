import argparse
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import gc
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

parser = argparse.ArgumentParser(description='Download and process tf files')
parser.add_argument('--saved_model_path', required=True,
                    help='path to saved model')
parser.add_argument('--test_path', required=True,
                    help='path to test image')

parser.add_argument('--output_path', required=True,
                    help='path to output predicted image')
parser.add_argument('--min_score_thresh', required=False, default=0.0,
                    help='min score threshold')
args = parser.parse_args()
PATH_TO_SAVED_MODEL = os.path.join(args.saved_model_path, "saved_model")
PATH_TO_TEST_IMAGE = args.test_path
PATH_TO_OUTPUT_IMAGE = args.output_path 
MIN_SCORE_THRESH = float(args.min_score_thresh)
os.makedirs(PATH_TO_OUTPUT_IMAGE, exist_ok=True)
# Load the Labels
PATH_TO_LABELS = "/target/waymo/code/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                        use_display_name=True)

# Load saved model and build the detection function
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

TOTAL_TEST_IMAGES = os.listdir(PATH_TO_TEST_IMAGE)
for filename in TOTAL_TEST_IMAGES:
    filename_path = os.path.join(PATH_TO_TEST_IMAGE, filename)
    print('Running inference for {}... '.format(filename_path))
    image_np = cv2.imread(filename_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=MIN_SCORE_THRESH,
          agnostic_mode=False)

    cv2.imwrite(os.path.join(PATH_TO_OUTPUT_IMAGE, filename), image_np_with_detections[:,:,::-1])
    print('Done')
    del image_np_with_detections
    gc.collect()
    


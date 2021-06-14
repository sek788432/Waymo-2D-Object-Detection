import argparse
import io
import os
import subprocess

# import ray
import tensorflow.compat.v1 as tf
from PIL import Image
from psutil import cpu_count

from utils import *
from object_detection.utils import dataset_util, label_map_util
import os
label_map = label_map_util.load_labelmap('/target/waymo/code/label_map.pbtxt')
label_map_dict = label_map_util.get_label_map_dict(label_map)

t2idict = {y:x for x,y in label_map_dict.items()}
def class_text_to_int(text):
    return t2idict[text]

def create_tf_example(filename, encoded_jpeg, annotations):
    """
    This function create a tf.train.Example from the Waymo frame.
    args:
        - filename [str]: name of the image
        - encoded_jpeg [bytes]: jpeg encoded image
        - annotations [protobuf object]: bboxes and classes
    returns:
        - tf_example [tf.Train.Example]: tf example in the objection detection api format.
    """

    # TODO: Implement function to convert the data
    encoded_jpg_io = io.BytesIO(encoded_jpeg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    image_format = b'jpeg'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for index, row in enumerate(annotations):
        
        xmin = row.box.center_x - row.box.length/2.0
        xmax = row.box.center_x + row.box.length/2.0
        ymin = row.box.center_y - row.box.width/2.0
        ymax = row.box.center_y + row.box.width/2.0
        
         
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(class_text_to_int(row.type).encode('utf8'))
        classes.append(row.type)
        print(class_text_to_int(row.type).encode('utf8'))
        print(row.type)

    filename = filename.encode('utf8')
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example

def process_tfr(filepath, output_dir):
    """
    process a Waymo tf record into a tf api tf record
    args:
        - filepath [str]: path to the Waymo tf record file
        - data_dir [str]: path to the destination directory
    """
    # create processed data dir

    dest = output_dir
    file_name = os.path.basename(filepath)
    logger = get_module_logger(__name__)
    if os.path.exists(f'{dest}/{file_name}'):
        return

    logger.info(f'Processing {filepath}')
    writer = tf.python_io.TFRecordWriter(f'{dest}/{file_name}')
    dataset = tf.data.TFRecordDataset(filepath, compression_type='')
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # FRONT 
        encoded_jpeg, annotations = parse_frame(frame, 'FRONT')
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
        # FORNT_LEFT
        encoded_jpeg, annotations = parse_frame(frame, 'FRONT_LEFT')
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
        # FRONT_RIGHT
        encoded_jpeg, annotations = parse_frame(frame, 'FRONT_RIGHT')
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
        # SIDE_LEFT
        encoded_jpeg, annotations = parse_frame(frame, 'SIDE_LEFT')
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
        # SIDE_RIGHT
        encoded_jpeg, annotations = parse_frame(frame, 'SIDE_RIGHT')
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()
    return

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--output_dir', required=True,
                        help='processed data directory')
    parser.add_argument('--filepath', required=True,
                        help='raw data path')
    args = parser.parse_args()
    process_tfr(args.filepath, args.output_dir)
    logger = get_module_logger(__name__)
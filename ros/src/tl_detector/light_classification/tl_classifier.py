# Written by Chris Ameter
# 2018-10-20
# Based on Tensorflow Object Detection Demo.
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

from styx_msgs.msg import TrafficLight

import numpy as np
import os
import tensorflow as tf

#from PIL import Image

#import matplotlib.pyplot as plt

import rospy

#from utils import label_map_util
#from utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self, is_site):
        # Load classifier
        SIM_MODEL = 'frozen_inference_graph_sim.pb'
        REAL_MODEL = 'frozen_inference_graph_real.pb'

        #PATH_TO_LABELS = 'label_map.pbtxt'
        #NUM_CLASSES = 14

        # Set path to frozen detection graph. This is the actual model that is used for the object detection.
        if is_site:
            path_to_frozen_graph = os.path.dirname(os.path.realpath(__file__)) + '/graphs/' +REAL_MODEL
        else:
            path_to_frozen_graph = os.path.dirname(os.path.realpath(__file__)) + '/graphs/' +SIM_MODEL

        # Load a (frozen) TensorFlow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Get handles to input and output tensors
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.return_tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                self.return_tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(key + ':0')

        # Create class index to TrafficLight message mapping
        # rosmsg show styx_msgs/TrafficLight
        # uint8 UNKNOWN=4
        # uint8 GREEN=2
        # uint8 YELLOW=1
        # uint8 RED=0
        self.TrafficLightClasstoMsgMap = [4, TrafficLight.GREEN, TrafficLight.RED, 4, 4, 4, 4, TrafficLight.YELLOW, 4, 4, 4, 4, 4, 4, 4] 

        # Load label map
        #Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine.
        #category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        rospy.logwarn("Getting classification...")

        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Run inference
                output_dict = sess.run(self.return_tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        state = set()
        prev_score = output_dict['detection_scores'][0]
        for i, score in enumerate(output_dict['detection_scores']):
            if score > 0.5 and prev_score - score < 0.1:
                state.add(output_dict['detection_classes'][i])
                prev_score = score
            else:
                break

        #self.output_debug(image, output_dict)

        if len(state) == 1:
            return self.TrafficLightClasstoMsgMap[state.pop()]
        else:
            return TrafficLight.UNKNOWN


    def output_debug(self, image, output_dict):
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]

        return

        category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 7: {'id': 7, 'name': 'Yellow'}, 8: {'id': 8, 'name': 'off'}}

        rospy.logwarn("output_dict: {}".format(output_dict))
        
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        plt.imshow(image)
        plt.show()



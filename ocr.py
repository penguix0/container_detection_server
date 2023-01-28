import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from detection import detection
from recognition import recognition
import numpy as np

config = {}
config['segm_conf_thr'] = 0.8
config['link_conf_thr'] = 0.8
config['min_area'] = 300
config['min_height'] = 10

def import_detection_model():
    with tf.Graph().as_default():
        detection_model_path = "./detection.pb"
        detection_graph_def = tf.GraphDef()
        with open(detection_model_path, "rb") as f:
            detection_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(detection_graph_def, name="")

        sess_d = tf.Session()
        init = tf.global_variables_initializer()
        print (init)
        sess_d.run(init)
        input_x = sess_d.graph.get_tensor_by_name("Placeholder:0")
        segm_logits = sess_d.graph.get_tensor_by_name("model/segm_logits/add:0")
        link_logits = sess_d.graph.get_tensor_by_name("model/link_logits/Reshape:0")

    return sess_d, input_x, segm_logits, link_logits

def import_recognition_h_model():
    with tf.Graph().as_default():
        recognition_model_h_path = "./recognition_h.pb"
        recogniton_graph_def = tf.GraphDef()
        with open(recognition_model_h_path, "rb") as f:
            recogniton_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(recogniton_graph_def, name="")

        sess_r_h = tf.Session()
        init = tf.global_variables_initializer()
        sess_r_h.run(init)
        input_h = sess_r_h.graph.get_tensor_by_name("Placeholder:0")
        #input_h = tf.placeholder(tf.float32, [4, 32, 240, 1])
        model_out_h = sess_r_h.graph.get_tensor_by_name("shadow/LSTMLayers/transpose:0")
        decoded_h, _ = tf.nn.ctc_beam_search_decoder(model_out_h, 60 * np.ones(4), merge_repeated=False)

        return sess_r_h, input_h, model_out_h, decoded_h

def import_recognition_v_model():
    with tf.Graph().as_default():
        recognition_model_v_path = "./recognition_v.pb"
        recogniton_graph_def = tf.GraphDef()
        with open(recognition_model_v_path, "rb") as f:
            recogniton_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(recogniton_graph_def, name="")

        sess_r_v = tf.Session()
        init = tf.global_variables_initializer()
        sess_r_v.run(init)
        input_v = sess_r_v.graph.get_tensor_by_name("Placeholder:0")
        #input_h = tf.placeholder(tf.float32, [4, 32, 240, 1])
        model_out_v = sess_r_v.graph.get_tensor_by_name("shadow/LSTMLayers/transpose:0")
        decoded_v, _ = tf.nn.ctc_beam_search_decoder(model_out_v, 60 * np.ones(4), merge_repeated=False)

        return sess_r_v, input_v, model_out_v, decoded_v

def sort_container_number(items):
    prefix = ""
    serial_number = ""
    check_digit = ""
    for item in items:
        if len(item) == 4 and item.isalpha():
            prefix = item
        elif item.isdigit() and len(item) == 7:
            serial_number = item
        elif item.isalnum() and len(item) == 4:
            check_digit = item
    if serial_number == "":
        for item in items:
            if item != prefix and item != check_digit:
                serial_number = item
    return [prefix, serial_number, check_digit]

sess_d, input_x, segm_logits, link_logits = import_detection_model()
sess_r_h, input_h, model_out_h, decoded_h = import_recognition_h_model()
sess_r_v, input_v, model_out_v, decoded_v = import_recognition_v_model()

import cv2

def recognize(image):
    bboxs = detection(image, sess_d, input_x, segm_logits, link_logits, config)

    if len(bboxs) == 0:
        return ""

    result = recognition(image, sess_r_h, sess_r_v , bboxs, (240, 32), input_h, input_v, model_out_h, model_out_v, decoded_h, decoded_v)
    
    result = sort_container_number(result)
    
    return result

import sys
sys.path.append('/home/firefly/.local/lib/python3.7/site-packages')
import platform
import os
import urllib
import traceback
import time
import datetime
import numpy as np
import cv2
from kafka import KafkaProducer
import json
from bson import json_util
from kafka.errors import KafkaError
from rknnlite.api import RKNNLite
import threading

RKNN_MODEL = './elenet_960_1280_yolov5s.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True

BOX_THRESH = 0.5
NMS_THRESH = 0.6
IMG_SIZE = 1280

CLASSES = ("electric_bicycle","person","door_sign","bicycle","gastank")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_confidences[...,0] >= BOX_THRESH)


    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
              [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

total_infer_times = 0
def process_frame_queue_for_infer():
    global frame_queue
    global frame_queue_lock
    global total_infer_times
    while (True):
        # print("dequeue thread is running")
        total_time_start = time.time()
        processing_frame = None
        try:
            frame_queue_lock.acquire()
            if frame_queue:
                # reduce the q
                processing_frame = frame_queue[-1]
                frame_queue = []
            else:
                time.sleep(1/100)
                continue
        except:
            continue
        finally:
            frame_queue_lock.release()

        resized_frame, ratio, (dw, dh) = letterbox(processing_frame, new_shape=(1280, 1280))
        # print('image shape: {}, ratio: {}'.format(resized_frame.shape, ratio))
        # # Inference
        print('  pre-processing used time: {}ms, frame_queue len: '.format((time.time() - total_time_start)*1000,len(frame_queue)))
        infer_start_time = time.time()
        outputs = rknn_lite.inference(inputs=[resized_frame])
        total_infer_times = total_infer_times+1
        print('  infer used time: {}ms'.format((time.time() - infer_start_time)*1000))

        post_process_time = time.time()
        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3,-1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3,-1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3,-1]+list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        boxes, classes, scores = yolov5_post_process(input_data)            
        detected_obj_count = 0
        detected_obj_names = ''
        if classes is not None:
            for cl in classes:
                detected_obj_names += CLASSES[cl]+";"
            detected_obj_count = len(classes)
        print('  post_process used time: {}ms, detected object names: {}'.format((time.time() - post_process_time)*1000,detected_obj_names))
        # enable_output_inferenced_image = False
        # if (detected_obj_count>=1 and enable_output_inferenced_image):
        #     for box, score, cl in zip(boxes, scores, classes):
        #         top, left, right, bottom = box
        #         print('class: {}, score: {}'.format(CLASSES[cl], score))
        #         # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        #         # top = int(top)
        #         # left = int(left)
        #         # right = int(right)
        #         # bottom = int(bottom)
        #         # img_1 = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
        #         img_1 = resized_frame
        #         if boxes is not None:
        #             draw(img_1, boxes, scores, classes)
        #             cv2.imwrite('./capture/{}.jpg'.format(datetime.datetime.now()), img_1)
        
        enable_output_eb_image = False
        if (detected_obj_count>=1):
            obj_info_list = []
            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box
                # sample    top: 585.4451804161072, left: 589.0, right: 1114.6779885292053, bottom: 915.0
                top = int(top)
                left = int(left)
                right = int(right)
                bottom = int(bottom)
                # CLASSES = ("electric_bicycle","person","door_sign","bicycle","gastank")
                if (cl==0):
                    cropped = resized_frame[top:top + int(bottom-top), left:left + int(right-left)]
                    frame_copy = cropped#cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)
                    print("see eb, resized_frame shape: {}, top: {}, left: {}, right: {}, bottom: {}".format(frame_copy.shape,top,left,right,bottom))
                    if frame_copy.shape[0]<=100 or frame_copy.shape[1]<=100:
                        print("!!!Too small eb image size, will skip...")
                        continue
                    if enable_output_eb_image:
                        img_1 = frame_copy
                        cv2.imwrite('./capture/eb_{}.jpg'.format(datetime.datetime.now()), img_1)
                    
                    retval, buffer = cv2.imencode('.jpg', frame_copy)
                    base64_bytes = base64.b64encode(buffer)
                    obj_base64_encoded_text = base64_bytes.decode('ascii')
                    eb_obj_info = '{}|{}|{}|{}|{}|Vehicle|#|TwoWheeler|B|M|b|X|base64_image_data:{}|{}'.format(18446744073709551615,top,left,right,bottom, 
                        obj_base64_encoded_text, score)
                    print("see eb, upload size: {} bytes".format(len(eb_obj_info)))
                    obj_info_list.append(eb_obj_info)
                if (cl==1):
                    people_obj_info = '{}|{}|{}|{}|{}|Person|#|m|18|b|n|f|{}'.format(18446744073709551615,top,left,right,bottom, score)
                    obj_info_list.append(people_obj_info)
                if (cl==2):
                    ds_obj_info = '{}|{}|{}|{}|{}|Vehicle|#|DoorWarningSign|B|M|y|l|CN|{}'.format(18446744073709551615,top,left,right,bottom, score)
                    obj_info_list.append(ds_obj_info)
                if (cl==3):
                    gt_obj_info = '{}|{}|{}|{}|{}|Vehicle|#|gastank|B|M|y|l|CN|{}'.format(18446744073709551615,top,left,right,bottom, score)
                    obj_info_list.append(gt_obj_info)
            if len(obj_info_list)>=1:
                producer.send(sensor_id_str, {
                    'version':'4.1',
                    'id':1913,
                    '@timestamp':'{}'.format(datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()),
                    'sensorId':'{}'.format(sensor_id_str),'objects':obj_info_list})
        print('Cycle used time: {}ms, total lasting time: {}s, total infered times: {}'.format((time.time() - total_time_start)*1000, time.time()-init_start_time, total_infer_times))

frame_queue_lock = threading.Lock()
frame_queue = []
if __name__ == '__main__':
    import argparse
    from json import dumps
    import configparser
    import base64
    parser = argparse.ArgumentParser()    
    parser.add_argument("-i", "--input-src-rtsp-uri", dest="input_src_rtsp_uri",
                      help="a rstp stream for start the inferencing",
                      default="rtsp://msg.glfiot.com:8554/mystream",
                      metavar="FILE")
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=False,
                        default="bicycletypenet_tao",
                        help='Name of model')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('--kafka-server-url',
                        type=str,
                        required=False,
                        default='msg.glfiot.com:9092',
                        help='kafka server URL. Default is xxx:9092.')
    FLAGS = parser.parse_args()
    if not os.path.exists('/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_elenet.txt'):
        print("Could not find config_elenet.txt under "
            "/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/")
        exit(-1)
    config = configparser.ConfigParser()
    config.read('/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_elenet.txt')
    sensor_id_str = config['custom-uploader']['whoami']

    # Create RKNN object
    rknn_lite = RKNNLite()

    if not os.path.exists(RKNN_MODEL):
        print('target rknn model: {} does not exist'.format(RKNN_MODEL))
        exit(-1)

    print('--> list devices:')
    rknn_lite.list_devices()
    print('done')

    print('--> query support target platform')
    rknn_lite.list_support_target_platform(rknn_model=RKNN_MODEL)
    print('done')
    print('--> Loading model')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # init runtime environment
    init_start_time = time.time()
    print('--> Init runtime environment')
    # run on RK3399Pro/RK1808 with Debian OS, do not need specify target.
    if platform.machine() == 'aarch64' or platform.machine() == 'armv7l':
        target = None
    else:
        target = 'rk1808'
    ret = rknn_lite.init_runtime(target='rv1126')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    print('  init_runtime used time: {}s'.format((time.time() - init_start_time)))

    dequeue_and_process_infer_thread = threading.Thread(target=process_frame_queue_for_infer, args=())
    dequeue_and_process_infer_thread.start()
    print("dequeue thread is started")
    while(True):
        try:
            producer = KafkaProducer(
                bootstrap_servers=FLAGS.kafka_server_url,
                value_serializer=lambda x: dumps(x).encode('utf-8'))
            producer.send(sensor_id_str, {"board-name":"rv1126-{}".format(sensor_id_str),
                "description":"board is online now, will infer src stream: {}".format(FLAGS.input_src_rtsp_uri)})
            # cap = cv2.VideoCapture("rtsp://admin:KSglfmis1@36.153.41.21:2121")
            cap = cv2.VideoCapture(FLAGS.input_src_rtsp_uri)
            print("Is VideoCapture opened: {}".format(cap.isOpened()))
            last_infered_time = None
            while(cap.isOpened()):
                # total_time_start = time.time()
                ret, frame = cap.read()
                # if last_infered_time is not None:
                #     if (time.time()- last_infered_time)<=1:
                #         # print("skip 1 frame...")
                #         continue
                # elif last_infered_time is None:
                #     continue
                try:
                    frame_queue_lock.acquire()
                    frame_queue.append(frame)
                finally:
                    frame_queue_lock.release()
        except Exception as e:
            print('exceptioned in process(conn to kafka server, open rtsp stream and infer): {} will keep retrying...'.format(traceback.format_exc()))
            time.sleep(1)
            continue


    print('quit the whole app')    
    rknn_lite.release()


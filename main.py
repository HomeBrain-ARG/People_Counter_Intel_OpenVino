"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os 
import sys
import time
import socket
import json
import cv2

import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from random import randint

# CPU extension for OpenVino v2019.R3.
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
GPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libclDNNPlugin.so" 

# COCO Labels:
LABELS_COCO = ["background","person", "bicycle", "car", "motorcycle", 
"airplane", "bus", "train", "truck", "boat", "traffic light", 
"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
"hair drier", "toothbrush"]

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# DEBUG MODE TURNED OFF by default, enable it if you need to debug the program (delete the # in the line below to activate it).
log.basicConfig(filename='log_people_counter.log', level=log.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    ### Argument descriptions:
    m_desc = "Path to an xml file with a trained model."
    i_desc = "Path to image or video file."
    l_desc = "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl."
    pt_desc = "Probability threshold for detections filtering (0.5 by default)."
    d_desc = "Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)."
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
### CHANGE 25/05/2020: Additional argument to activate/deactivate stats. ###
    s_desc = "Flag indicating whether to publish statistics to the MQTT client (activated by default)."

    parser.add_argument("-m", "--model", required=True, type=str, help=m_desc)
    parser.add_argument("-i", "--input", required=True, type=str, help=i_desc)
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None, help=l_desc)
    parser.add_argument("-d", "--device", type=str, default="CPU", help=d_desc)
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help=pt_desc)
    parser.add_argument("-c", "--color", type=str, default='BLUE', help=c_desc)
### CHANGE 25/05/2020: Additional argument to activate/deactivate stats. ###    
    parser.add_argument("-s", "--publish_stats", type=bool, default=True, help=s_desc)

    log.info("Main - build_argparser(): executed.")        
    return parser

def connect_mqtt(publish_stats):
    ### TODO: Connect to the MQTT client ###
    client = None
    if publish_stats:
        client = mqtt.Client()
        client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
   
    log.info("Main - connect_mqtt(): executed.")            
    return client

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        log.info("Main - convert_color(): User selected color executed.")            
        return out_color
    else:
        log.info("Main - convert_color(): Default color executed.")
        return colors['BLUE']

def draw_boxes(frame, output_network, args, width, height, inf_time):
    '''
    Draw bounding boxes onto the frame.
    '''
    ### Rectangle graphical configuration.
    rec_color       = convert_color(args.color)
    rec_thickness   = 1
    rec_linetype    = cv2.LINE_AA
    ### Configure inference time text configuration:
    inf_time_msg    = ''
    font            = cv2.FONT_HERSHEY_SIMPLEX 
    org             = (5, 15) 
    fontScale       = 0.5
    color           = (0, 0, 255) # Red color in BGR.
    thickness       = 1
    ### Set other variables:
    conf_percent = 0

    for box in output_network[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), rec_color, rec_thickness, rec_linetype)

    ### CHANGE 24/05/2020: I added the label and probability of detection. ###
            # Draw detection label:
            class_idx       = int(box[1])
            xmin_pos_text   = (xmin + 5)
            ymax_pos_text   = (ymax - 5)

            if 0 <= class_idx < len(LABELS_COCO):
                class_name = LABELS_COCO[class_idx]
            else:
                class_name = ''
                conf_percent = (conf * 100)
            cv2.putText(frame, "{} {:.2%}".format(class_name, conf), (xmin_pos_text, ymax_pos_text), font, fontScale, rec_color, thickness, cv2.LINE_AA)

    ### CHANGE 25/05/2020: I move the inference time text to this function. ###
        # Write inference time in the frame.
        inf_time_msg = "Inference time: {:.2f}ms.".format(inf_time * 1000)
        cv2.putText(frame, inf_time_msg, org, font, fontScale, color, thickness, cv2.LINE_AA) 

        log.info("Main - draw_boxes(): Executed.")
        log.info(frame.shape)

    return frame

### CHANGE 25/05/2020: Function get_statistics() to count persons and times. ###
def get_statistics(result, n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, last_detection_timestamp, flag_alert_when_person_leaves, video_timestamp, prob_threshold):
    '''
    Get statistics regarding people on screen, duration they spend on screen, and total people counted.
    '''
    TIME_BUFFER = 1.2
    
    # Return time on screen when a person has left the scene to update average duration.
    time_person_on_screen = None
    
    # Get frame's first object detected info.
    detections                  = result[0][0] # Output shape is 1x1x100x7
    first_detection             = detections[0]
    first_detection_conf        = first_detection[2]
    first_detection_class_idx   = int(first_detection[1])

    if 0 <= first_detection_class_idx < len(LABELS_COCO):
        first_detection_class_name = LABELS_COCO[first_detection_class_idx]
    else:
        first_detection_class_name = ''
    
    # Update time since last detection.
    time_since_last_detection = (video_timestamp - last_detection_timestamp)
    
    # If person detected in the frame with confidence.
    if first_detection_conf >= prob_threshold and first_detection_class_name == 'person':
    
        # If there hasn't been a detection in more than a second, confirm person entered the scene.
        if time_since_last_detection >= TIME_BUFFER:
            n_in_frame                  = 1
            n_persons_entered           += 1
            timestamp_person_entered    = video_timestamp
            # Turn on alert to notify when person will leave the scene.
            flag_alert_when_person_leaves = True
        
        # Return new detection timestamp.
        return n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, time_person_on_screen, video_timestamp, flag_alert_when_person_leaves
    
    # If no person detected in the frame.
    else:
        
        # If there hasn't been a detection in more than a TIME_BUFFER time, confirm person left the scene about a TIME_BUFFER time ago.
        if time_since_last_detection >= TIME_BUFFER and flag_alert_when_person_leaves:
            n_in_frame              = 0
            n_persons_left          += 1
            time_person_on_screen   = ((video_timestamp - TIME_BUFFER) - timestamp_person_entered)
            # Mute alert for person leaving until new person detected. 
            flag_alert_when_person_leaves = False
        
        # Return the last detection timestamp until new detection.
        return n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, time_person_on_screen, last_detection_timestamp, flag_alert_when_person_leaves

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Set argument variables.
    model           = args.model
    cpu_ext         = args.cpu_extension
    device          = args.device
    prob_threshold  = args.prob_threshold
    color           = args.color

    # Flag of selection of input image or video.
    single_image_mode = False

    # iniatilize variables:
    cur_request_id  = 0
    time_start      = 0
    time_end        = 0
    inf_time        = 0
    alarm = ["Lot of visits", "alarm 1", "alarm 2"]

    # iniatilize variables:
    frame_counter = 0
    n_in_frame = 0
    n_persons_entered = 0
    n_persons_left = 0
    timestamp_person_entered = 0
    time_people_on_screen = 0
    last_detection_timestamp = 0
    flag_alert_when_person_leaves = False    

    # Connect to the MQTT server.
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

   # Convert the args for color and probability.
    prob_threshold = float(prob_threshold)

    log.info("Main - infer_on_stream(): Initialized variables.")

    # Initialise the class
    infer_network = Network()
    log.info("Main - infer_on_stream(): Network class initialized.")
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, device, CPU_EXTENSION, cur_request_id)
    net_input_shape = infer_network.get_input_shape()
    log.info("Main - infer_on_stream(): Model loaded.")

    ### TODO: Handle the input stream ###
### CHANGE 24/05/2020: I added file check, keep in mind that additionally I have changed the order (CAM, Video and Image). ###
    # Webcam:
    if args.input == 'CAM':
        input_camvidimg = 0
        log.info("Main - infer_on_stream(): WEBCAM mode selected.")
    # Video:
    elif args.input.endswith('.mp4') or args.input.endswith('.avi'):
        input_camvidimg = args.input
        log.info("Main - infer_on_stream(): VIDEO mode selected.")
    # Image:
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_camvidimg = args.input
        log.info("Main - infer_on_stream(): IMAGE mode selected.")
    else:
        log.error("Main - infer_on_stream [ERROR]: File doesn't exist.")
        print("Main - infer_on_stream [ERROR]: Please enter a corect file.")
        sys.exit(1)

    cap = cv2.VideoCapture(input_camvidimg)
    cap.open(input_camvidimg)
    log.info("Main - infer_on_stream(): CV2 VideoCapture initialized.")

    # Grab the shape of the input.
    width = int(cap.get(3))
    height = int(cap.get(4))
### CHANGE 25/05/2020: Calculate FPS of input stream in webcam, video or image. ###
    # Obtain FPS from input stream.
    CAP_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    log.info("Main - infer_on_stream(): CV2 obtaining FPS.")

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        log.info("Main - infer_on_stream(): Entering in stream loop.")
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        frame_counter += 1
        video_timestamp = (frame_counter / CAP_FPS)
        log.info("Main - infer_on_stream(): Calculate video timestamp.")

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        log.info("Main - infer_on_stream(): Frame preprocessed.")

        ### TODO: Start asynchronous inference for specified request ###
        # Start measure of inference.
        time_start = time.time()
        log.info("Main - infer_on_stream(): Read inference start time.")
        # Perform inference on the frame.
        infer_network.exec_net(p_frame, cur_request_id) 
        log.info("Main - infer_on_stream(): Performing frame inference.")

        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            ### TODO: Get the results of the inference request ###
            output_network = infer_network.get_output(cur_request_id)
            log.info("Main - infer_on_stream(): Get results using get_output.")
        # Stop measure of inference.
        time_end = time.time()
        log.info("Main - infer_on_stream(): Read inference stop time.")

        # Calculate the inference time.
        inf_time = (time_end - time_start)
        log.info("Main - infer_on_stream(): Calculate inference time.")

        # Update the frame to include detected bounding boxes.
        frame = draw_boxes(frame, output_network, args, width, height, inf_time)
        log.info("Main - infer_on_stream(): Drawing boxes, object detection name and inference time.")

        ### TODO: Extract any desired stats from the results ###
        n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, time_person_on_screen, last_detection_timestamp, flag_alert_when_person_leaves = get_statistics(output_network, n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, last_detection_timestamp, flag_alert_when_person_leaves, video_timestamp, prob_threshold)
        log.info("Main - infer_on_stream(): Obtain statistics.")

### CHANGE 25/05/2020: New function to publish the messages in MQTT server. ###
        ### TODO: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        # If argument publish_stats is True, send data to MQTT server:
        if args.publish_stats:
            client.publish("person", json.dumps({"count": n_in_frame, "total": n_persons_entered}))
            # If another person left the scene.
            if time_person_on_screen:
                # Calculate the new average duration on scene.
                time_people_on_screen += time_person_on_screen
                time_people_on_screen_avg = time_people_on_screen / n_persons_left
                client.publish("person/duration", json.dumps({"duration": time_people_on_screen_avg}))
            log.info("Main - infer_on_stream(): Data sent to MQTT server.")

        # Print a congratulations message if your receive at least 5 persons (I was change the message accordingly to the current times, hehehe):
        if n_persons_entered >= 5:
            people_counter_msg1 = "You're a lucky person, in pandemic times, "
            people_counter_msg2 = "you have been received 5 visits :-)!"
            ### Configure inference time text in screen:
            font        = cv2.FONT_HERSHEY_SIMPLEX 
            org1        = (150, 400) 
            org2        = (185, 420)
            fontScale   = 0.7
            color       = (255, 255, 255) # Red color in BGR.
            thickness   = 1
            cv2.putText(frame, people_counter_msg1, org1, font, fontScale, color, thickness, cv2.LINE_AA) 
            cv2.putText(frame, people_counter_msg2, org2, font, fontScale, color, thickness, cv2.LINE_AA) 
            # Software publishes a message through MQTT when it detects at least 5 persons (needs changes in the UI):
            client.publish("person", json.dumps({"alarms": alarm[0]}))
        log.info("Main - infer_on_stream(): Writing statistics in MQTT server.")

        if key_pressed == 27:
            break

        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        log.info("Main - infer_on_stream(): Frame sent to FFMPEG server.")        
        
        # Write video for test purposes.
        #fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        #test_out_video = cv2.VideoWriter('test_out_video.vid', fourcc, 30, (width,height))

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode == True:
            cv2.imwrite('output_image.jpg', frame)
        log.info("Main - infer_on_stream(): Single_image_mode.")
    ### END OF THE INFERENCE.
    log.info("Main - infer_on_stream(): End of inference.")
    
    ### Release recurses:
    # Release the capture and destroy any OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT.
    if args.publish_stats:
        client.disconnect()
    log.info("Main - infer_on_stream(): Recurses released.")

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    log.info("main(): Build argparser.")
    # Connect to the MQTT server
    client = connect_mqtt(args.publish_stats)
    log.info("main(): MQTT connected.")
    # Perform inference on the input stream
    infer_on_stream(args, client)
    log.info("main(): Performing inference.")    

if __name__ == '__main__':
    main()

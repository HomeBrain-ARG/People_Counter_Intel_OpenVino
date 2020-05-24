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

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Only enable if you need to debug the program.
#log.basicConfig(filename='log_people_counter.log', level=log.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

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

    parser.add_argument("-m", "--model", required=True, type=str, help=m_desc)
    parser.add_argument("-i", "--input", required=True, type=str, help=i_desc)
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None, help=l_desc)
    parser.add_argument("-d", "--device", type=str, default="CPU", help=d_desc)
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help=pt_desc)
    parser.add_argument("-c", "--color", type=str, default='BLUE', help=c_desc)

    log.info("Main - build_argparser(): executed.")        
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    #client = None
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
        log.info("Main - convert_color(): user selected color executed.")            
        return out_color
    else:
        log.info("Main - convert_color(): default color executed.")
        return colors['BLUE']

def draw_boxes(frame, output_network, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    # Reset people_counter.
    people_counter = 0

    # Rectangle graphical configuration.
    rec_color       = convert_color(args.color)
    rec_thickness   = 1
    rec_linetype    = cv2.LINE_AA

    for box in output_network[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), rec_color, rec_thickness, rec_linetype)

            # When draw the box counts +1 person.
            people_counter += 1
    
    log.info("Main - draw_boxes(): executed.")
    return frame, people_counter

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

    # iniatilize variables: VER CUALES SIRVEN!!!!
    last_count = 0
    total_count = 0
    start_inf_time = 0
    end_inf_time = 0
    alarm = ["Lot of visits", "alarm 1", "alarm 2"]

    # Connect to the MQTT server.
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

   # Convert the args for color and probability.
    prob_threshold = float(prob_threshold)

    log.info("Main - infer_on_stream(): initialized variables.")

    # Initialise the class
    infer_network = Network()
    log.info("Main - infer_on_stream(): Network class initialized.")
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, device, GPU_EXTENSION)
    net_input_shape = infer_network.get_input_shape()
    log.info("Main - infer_on_stream(): model loaded.")

    ### TODO: Handle the input stream ###
    # Webcam:
    if args.input == 'CAM':
        input_camvidimg = 0
        log.info("Main - infer_on_stream(): webcam mode selected.")
    # Image:
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_camvidimg = args.input
        log.info("Main - infer_on_stream(): image mode selected.")
    # Video:
    else:
        input_camvidimg = args.input
        log.info("Main - infer_on_stream(): video mode selected.")
        assert os.path.isfile(args.input), "Main - infer_on_stream [ERROR]: File doesn't exist."

    cap = cv2.VideoCapture(input_camvidimg)
    cap.open(input_camvidimg)
    log.info("Main - infer_on_stream(): CV2 VideoCapture initialized.")

    # Grab the shape of the input.
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        log.info("Main - infer_on_stream(): entering in while.")
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        log.info("Main - infer_on_stream(): frame preprocessed.")

        ### TODO: Start asynchronous inference for specified request ###
        # Counter to measure inference.
        start_inf_time = time.time()
        # Perform inference on the frame.
        infer_network.exec_net(p_frame) 
        log.info("Main - infer_on_stream(): performing frame inference.")

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            end_inf_time = time.time() - start_inf_time

            ### TODO: Get the results of the inference request ###
            output_network = infer_network.get_output()
            log.info("Main - infer_on_stream(): get results using get_output.")
            # Update the frame to include detected bounding boxes.
            frame, people_counter = draw_boxes(frame, output_network, args, width, height)
            log.info("Main - infer_on_stream(): drawing boxes.")
            # Write a message in the frame.
            end_inf_time_msg = "Inference time: {:.2f}ms.".format(end_inf_time * 1000)

            ### Configure inference time text in screen:
            font        = cv2.FONT_HERSHEY_SIMPLEX 
            org         = (5, 15) 
            fontScale   = 0.5
            color       = (0, 0, 255) # Red color in BGR.
            thickness   = 1
            cv2.putText(frame, end_inf_time_msg, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
            log.info("Main - infer_on_stream(): writing inference time.")

            ### TODO: Extract any desired stats from the results ###
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # Total persons counted:
            if people_counter > last_count:
                start_time = time.time()
                total_count = (total_count + people_counter - last_count)
                client.publish("person", json.dumps({"total": total_count}))

            # Duration of persons in the frame:
            if people_counter < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": people_counter}))
            last_count = people_counter

            # Print a congratulations message if your receive at least 5 persons:
            if total_count >= 5:
                people_counter_msg = "You're a lucky person, you have been received 5 visits!!"
                
                ### Configure inference time text in screen:
                font        = cv2.FONT_HERSHEY_SIMPLEX 
                org         = (25, 420) 
                fontScale   = 0.8
                color       = (255, 255, 255) # Red color in BGR.
                thickness   = 2
                cv2.putText(frame, people_counter_msg, org, font, fontScale, color, thickness, cv2.LINE_AA) 
                
                # Software publishes a message through MQTT when it detects at least 5 persons (needs changes in the UI):
                client.publish("person", json.dumps({"alarms": alarm[0]}))
            log.info("Main - infer_on_stream(): writing statistics.")

            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        log.info("Main - infer_on_stream(): frame sent to FFMPEG server.")        

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode == True:
            cv2.imwrite('output_image.jpg', frame)
        log.info("Main - infer_on_stream(): single_image_mode.")
    ### END OF THE INFERENCE.
    log.info("Main - infer_on_stream(): End of inference.")
    
    ### Release recurses:
    # Release the capture and destroy any OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT.
    client.disconnect()
    log.info("Main - infer_on_stream(): release recurses.")

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    log.info("main(): Build argparser.")
    # Connect to the MQTT server
    client = connect_mqtt()
    log.info("main(): MQTT connected.")
    # Perform inference on the input stream
    infer_on_stream(args, client)
    log.info("main(): Performing inference.")    

if __name__ == '__main__':
    main()

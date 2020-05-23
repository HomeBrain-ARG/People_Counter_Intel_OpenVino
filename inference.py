#!/usr/bin/env python3
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices and performs synchronous and asynchronous modes for the specified infer requests.
    """
    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device, cpu_extension, num_requests=0):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        log.info("Inference - load_model(): Model loaded.")
        # Initialize the plugin.
        self.plugin = IECore()
        log.info("Inference - load_model(): Initialized plugin.")

        ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        log.info("Inference - load_model(): Extension added.")
        # Read the IR as a network.
        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("Inference - load_model(): IR was read.")
        ### TODO: Check for supported layers ###
        if device == "CPU":
            supported_layers = self.plugin.query_network(self.network, device)
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("Inference - load_model() [ERROR]: Unsupported layers found {}.".format(unsupported_layers))
                sys.exit(1)

        # Load the IENetwork into the plugin.
        if num_requests == 0:
            self.exec_network = self.plugin.load_network(self.network, device)
            log.info("Inference - load_model(): IENetwork loaded and num_requests=0.")
        else:
            self.exec_network = self.plugin.load_network(self.network, device, num_requests=num_requests)
            log.info("Inference - load_model(): IENetwork loaded and num_requests=num_requests.")
        # Get the input and output layers.
        self.input_blob = next(iter(self.network.inputs))
        log.info("Inference - load_model(): Input blob initialized.")
        self.output_blob = next(iter(self.network.outputs))
        log.info("Inference - load_model(): Output blob initialized.")

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        '''
        Gets the input shape of the network.
        '''
        ### TODO: Return the shape of the input layer ###
        log.info("Inference - get_input_shape(): executed.")
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        '''
        Given an input image, this function makes an asynchronous inference request.
        '''
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        log.info("Inference - exec_net(): executed.")
        return

    def wait(self):
        '''
        Function to check the status of the inference request.
        '''
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        wait_status = self.exec_network.requests[0].wait(-1)
        log.info("Inference - wait(): executed.")
        return wait_status

    def get_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        log.info("Inference: get_output executed.")        
        return self.exec_network.requests[0].outputs[self.output_blob]
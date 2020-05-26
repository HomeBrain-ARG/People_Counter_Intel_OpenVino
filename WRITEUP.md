# Intel® Edge AI for IoT Developers Nanodegree Program

# Project N°1 - Write-Up:


To develop the present project, tests were carried out with three pre-trained TensorFlow models and the results obtained were compared against a pre-trained Intel® model. In summary, I can said that much better results were obtained with the Intel® pre-trained model, I tested the models on CPU and GPU.


## Explaining Custom Layers:

Custom Layers conversion flow is as follows:


![Custom Layers Conversion Flow](./images/workflow_steps.png)

Link: https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html


The tool used to perform the conversions of different pre-trained models of the TF, Caffe and ONNX type, among others to OPenVino is called "Model Optimizer".

In general this tool, made up of several scripts, is found in the path: ```/opt/intel/openvino/deployment_tools/model_optimizer```.

The "Model Optimizer" produces an "Intermediate Representation" or "IR" of the network, which can be read, loaded, and inferred with the "Inference Engine" as we see in the previous graphic. 

After the conversion, IR will be made up of two files:
* File with **.xml** extension: Describes the network topology.
* File with **.bin** extension: Contains the weights and biases binary data.

The mechanics to perform a conversion is as follows:

• Using Model Optimizer to generate IR files containing the custom layer.
• Edit the CPU Extension Template Files.
• Execute the model with the custom layer.

In our project, pre-trained SSD TensorFlow models have been used, as we have observed, the more complex the chosen models have been, the more custom layers we have found.

We'll see an example of conversion and executing using the pre-trained model called **SSD_MobileNet_v1_coco_2018_01_28**:



## Comparing Model Performance:

Como se ha indicado previamente, se han seleccionado modelos de TensorFlow de la siguiente página: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.

Los seleccionados para realizar las pruebas fueron los siguientes:

1) ssd_mobilenet_v1_coco_2018_01_28: 
    - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz.

2) ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03: 
    - http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz.

3) ssd_mobilenet_v2_coco_2018_03_29: 
    - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz.

These models were compared to the Intel model, downloaded directly with the script "\opt\intel\openvino\deployment_tools\tools\model_downloader\downloader.py".

4) pedestrian-detection-adas-0002: 
        - https://docs.openvinotoolkit.org/latest/_models_intel_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html.

### Models comparison:

#### _Using GPU Intel Corporation HD Graphics 530:_
| Model | Size before conversion [MB] | Size after conversion [MB] | Average inference Speed [ms] | Memory Usage [MB] |
|:-----:|:---------------------------:|:--------------------------:|:----------------------------:|:-----------------:|
ssd_mobilenet_v1_coco | 59.5 | 27.3 | 37.18 | 266,9
ssd_resnet50_v1 | 270.2 | 206.9 | 46.35 | 266.4
ssd_mobilenet_v2_coco | 140.8 | 67.4 | 32.62 | 266.7
pedestrian-detection-adas-0002 | - | 4.7 | 28.23 | 267.0

#### _Using CPU Intel Core i7-6700K:_
| Model | Size before conversion [MB] | Size after conversion [MB] | Average inference Speed [ms] | Memory Usage [MB] |
|:-----:|:---------------------------:|:--------------------------:|:----------------------------:|:-----------------:|
ssd_mobilenet_v1_coco | 59.5 | 27.3 | 35.63 | 114.3 
ssd_resnet50_v1 | 270.2 | 206.9 | 441.32 | 554.2
ssd_mobilenet_v2_coco | 140.8 | 67.4 | 28.26 | 198.4
pedestrian-detection-adas-0002 | - | 4.7 | 24.3 | 82.7

All models have been tested with probability thresholds of: 0.2; 0.3; 0.5; 0.8 and 0.95.

It is observed that the most efficient model ends up being the "Pedestrian-detection-adas-0002" embedded within the algorithm that includes the OpenVino v2019.R3 toolkit.


## Assess Model Use Cases:

In my case, because I work in industrial environments, I find it very useful to use this type of tools in personal safety systems, such as: detection of personnel working in areas with movements of systems that move loads (cranes, for instance) on top of them; in this case, the advance of the same towards where the personnel are could be disabled by the automation system, preventing incidents and/or accidents.


## Assess Effects on End User Needs:

In vision systems in industrial environments, the lighting issue is very complex. In general, it's sought to install light reflectors where video cameras are installed, this in order to avoid errors due to adverse environmental conditions.

With reference to the accuracy of the models, I consider that to detect people, the examples within the OpenVino toolkit are sufficient, in industrial environments the existing models are generally scarce. This is why you have to think about training models from the beginning of a project and then use them with OpenVino.

## Model Research:

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v1_coco.
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments: "python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json".
  - The model was insufficient for the application because the precision is not correct, it does not continually detect people.
  - I tried to improve the model for the app by modifying the probability thresholds.
  
- Model 2: ssd_resnet50_v1_fpn.
  - http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz.
  - I converted the model to an Intermediate Representation with the following arguments: "python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json".
  - The model was insufficient for the application because the precision is not correct, it does not continually detect people. 
  - I tried to improve the model for the app by modifying the probability thresholds.

- Model 3: ssd_mobilenet_v2_coco.
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz.
  - I converted the model to an Intermediate Representation with the following arguments: "python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json".
  - The  model was insufficient for the application because the precision is not correct, it does not continually detect people.
  - I tried to improve the model for the app by modifying the probability thresholds.

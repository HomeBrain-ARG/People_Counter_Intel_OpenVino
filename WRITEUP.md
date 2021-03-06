# Intel® Edge AI for IoT Developers Nanodegree Program

# Project N°1 - Write-Up:


To develop the present project, tests were carried out with three pre-trained TensorFlow models and the results obtained were compared against a pre-trained Intel® model. In summary, I can said that much better results were obtained with the Intel® pre-trained model, I tested the models on CPU and GPU.

The basic execution of the app is as follows:
```
python3 main.py -i [CAM|Video|Image] -m [file of model in .xml format] -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d [GPU|CPU|MYRIAD] --prob_threshold [values from 0.0 to 1.0, recommended 0.3 to 0.7] --color [BLUE|GREEN|RED] | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

Execution example:
```
python3 main.py -i Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --prob_threshold 0.3 --color GREEN | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

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
• Execute the model with the custom layer.

In our project, pre-trained SSD TensorFlow models have been used, as we have observed, the more complex the chosen models have been, the more custom layers we have found.

We'll see an example of conversion and executing using the pre-trained model of TF called **SSD_MobileNet_v1_coco_2018_01_28**:

### **First step - Conversion:**
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json
```

In this step keep in mind the different parameters depending on the type of pre-trained model, for instance: in this case, TF models need the parameter **--reverse_input_channels** because, as the parameter said, the channels are inverted with respect to OpenVino models.

After this step I obtained the following files in the same directory where the original model is located:

- frozen_inference_graph.bin
- frozen_inference_graph.xml

These two files are the ones that we'll need later to run our app.


### **Send step - Execution:**

Execution of the app using the example file "Pedestrian_Detect_2_1_1.mp4":
```
python3 main.py -i Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --prob_threshold 0.3 --color GREEN | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

Another way to execute the same app using a WEBCAM is the following:
```
python3 main.py -i CAM -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --prob_threshold 0.3 --color GREEN | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```


### **Obtaining help using the app:**

To obtain more details about the execution of the app, please execute the following in the command line:
```
python3 main.py --help
```

Or:
```
python3 main.py -h
```
Typing this will run the program's help.


## Comparing Model Performance:

As previously indicated, TensorFlow models have been selected from the following page: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.

Those selected to perform the tests were the following:

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
| ssd_mobilenet_v1_coco | 59.5 | 27.3 | 37.18 | 266,9 |
| ssd_resnet50_v1 | 270.2 | 206.9 | 46.35 | 266.4 |
| ssd_mobilenet_v2_coco | 140.8 | 67.4 | 32.62 | 266.7 |
| pedestrian-detection-adas-0002 | - | 4.7 | 28.23 | 267.0 |

#### _Using CPU Intel Core i7-6700K:_
| Model | Size before conversion [MB] | Size after conversion [MB] | Average inference Speed [ms] | Memory Usage [MB] |
|:-----:|:---------------------------:|:--------------------------:|:----------------------------:|:-----------------:|
| ssd_mobilenet_v1_coco | 59.5 | 27.3 | 35.63 | 114.3 |
| ssd_resnet50_v1 | 270.2 | 206.9 | 441.32 | 554.2 |
| ssd_mobilenet_v2_coco | 140.8 | 67.4 | 28.26 | 198.4 |
| pedestrian-detection-adas-0002 | - | 4.7 | 24.3 | 82.7 |

All models have been tested with probability thresholds of: 0.2; 0.3; 0.5; 0.8 and 0.95.

It is observed that the most efficient model ends up being the "Pedestrian-detection-adas-0002" embedded within the algorithm that includes the OpenVino v2019.R3 toolkit.


## Assess Model Use Cases:

In my case, because I work in industrial environments, I find it very useful to use this type of tools in personal safety systems, such as: detection of personnel working in areas with movements of systems that move loads (cranes, for instance) on top of them; in this case, the advance of the same towards where the personnel are could be disabled by the automation system, preventing incidents and/or accidents.


With the same reasoning as above, this app can be used additionally to detect people on the threshold of a selected place, for example, serving to notify your cell phone that someone is in a certain place.

The last example can be the use of this system as it is to count the average time that a person is in front of a computer, with any webcam (including the one included in their notebook) this action can be performed.
This is because within the variables it reports, the average time in front of the camera is found.

For the last examples, you only need to run the following command to detect people with a high degree of precision:
```
python3 main.py -i CAM -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --prob_threshold 0.3 --color GREEN | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Assess Effects on End User Needs:

* In vision systems in industrial environments, the lighting issue is very complex. In general, it's sought to install light reflectors where video cameras are installed, this in order to avoid errors due to adverse environmental conditions.
With reference to the accuracy of the models, I consider that to detect people, the examples within the OpenVino toolkit are sufficient, in industrial environments the existing models are generally scarce. This is why you have to think about training models from the beginning of a project and then use them with OpenVino.
We must understand that the lower the illumination, the lower the model's accuracy will be.

* The focal length is part of the intrinsic characteristics of a camera, for applications where the distances vary, it is possible to use motorized lenses to correct this variable. In this way you will always have an acquisition of the image with good precision.
Keep in mind that, in the case of a Webcam, the focal lengths do not exceed tens of millimeters (for 1080p, in general, they are between 3mm and 12mm approx.), This means that they can capture images at a maximum distances of between 7m to 20m.

* The size of the image is another intrinsic condition of the construction of the cameras, we can say that the larger the image, the more elements in the frame can be covered, but with smaller objects. With a smaller image we should have a better focus.
Apart from this, it is necessary to select the model according to the image size, otherwise we may have false positives or a decrease in accuracy when the model detects the elements.

## Model Research:

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v1_coco.
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments: 
  ```
  python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json
  ```
  - The converted model worked in efficient way with a good precision.
  - I tried differents probability thresholds from 0.3 to 0.8 with success. The recommended value is 0.3.
  
- Model 2: ssd_resnet50_v1_fpn.
  - http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz.
  - I converted the model to an Intermediate Representation with the following arguments: 
  ```
  python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The converted model doesn't worked in efficient way because the low speed of inference, approx. 500ms per frame. At the end of the day the accuracy was acceptable. 
  - The recommended value of probability threshold was 0.3.

- Model 3: ssd_mobilenet_v2_coco.
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz.
  - I converted the model to an Intermediate Representation with the following arguments: 
  ```
  python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The converted model worked in efficient way (32ms of iference time per frame) with a good precision.
  - I tried differents probability thresholds from 0.3 to 0.8 with success. The recommended value is 0.3.

- Model 4: pedestrian-detection-adas-0002: 
    - https://docs.openvinotoolkit.org/latest/_models_intel_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html.
  - This model doesn't need conversion because is and Intel model compatible with OpenVino (.xml and .bin files are provided directly).
  - You can execute directly the app using this model as follows:
  ```
  python3 main.py -i Pedestrian_Detect_2_1_1.mp4 -m pedestrian-detection-adas-0002.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --prob_threshold 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
  - The converted model was very efficient (25ms of iference time per frame) with a very good accuracy (over 80%).
  - I tried different probability thresholds from 0.3 to 0.9. Because the model is overtrained I don't recommend probability thresholds less than 0.7, otherwise the model generates false positives.

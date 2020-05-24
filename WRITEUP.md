# Intel® Edge AI for IoT Developers Nanodegree Program

# Project N°1 - Write-Up:


To develop the present project, tests were carried out with three pre-trained TensorFlow models and the results obtained were compared against a pre-trained Intel® model. In summary, I can said that much better results were obtained with the Intel® pre-trained model, I tested the models on CPU and GPU.


## Explaining Custom Layers:

In our project, pre-trained TensorFlow models have been used, as we have observed, the more complex the chosen models have been, the more custom layers we have found.

Due to the complexity of the process, I have carefully followed the steps indicated in the documentation of OpenVino v2019.R3:

https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Offloading_Sub_Graph_Inference.html

Una de las tareas que quise realizar es, detectar adicionalmente el género de cada persona que aparecía de espaldas en la cámara, en el video original del proyecto "Pedestrian_Detect_2_1_1.mp4". Para esto quise utilizar un modelo Inception V3 de TensorFlow/Keras: 

One of the tasks I wanted to carry out, is to additionally detect the gender of each person who appeared with his back to the camera, in the original video of the project "Pedestrian_Detect_2_1_1.mp4". For this I wanted to use a TensorFlow / Keras Inception V3 model: https://github.com/tensorflow/models/tree/master/research/inception

For this I used the following pages to better understand if it was going to be useful to me:
https://github.com/scoliann/GenderClassifierCNN
https://cloud.google.com/tpu/docs/inception-v3-advanced

Unfortunately I could not generate the IR of Inception V3 with the Model Optimizer because it gave me countless errors. I had very little time to finish Project No. 1 and decided to use simpler models to the implementation (mainly SSD pre-trained models).

Another model that was tried to be used was:

* faster_rcnn_nas_coco_2018_01_28:
    - http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz

In this case I was able to convert it, for a time I dismissed it due to its large size and slowness.

## Comparing Model Performance:

Como se ha indicado previamente, se han seleccionado modelos de TensorFlow de la siguiente página: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.

Los seleccionados para realizar las pruebas fueron los siguientes:

1) ssd_mobilenet_v1_coco_2018_01_28: 
    - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz.

2) ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03: 
    - http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

3) ssd_mobilenet_v2_coco_2018_03_29: 
    - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

These models were compared to the Intel model, downloaded directly with the script "\opt\intel\openvino\deployment_tools\tools\model_downloader\downloader.py".

4) pedestrian-detection-adas-0002: 
        - https://docs.openvinotoolkit.org/latest/_models_intel_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html

### Models comparison:

### Using GPU Intel Corporation HD Graphics 530:
Model | Size before conversion [MB] | Size after conversion [MB] | Average inference Speed [ms] | Memory Usage [MB]
------|----------------------------|-----------------------------|------------------------------|------------------
ssd_mobilenet_v1_coco | 59.5 | 27.3 | 37.18 | 266,9
ssd_resnet50_v1 | 270.2 | 206.9 | 46.35 | 266.4
ssd_mobilenet_v2_coco | 140.8 | 67.4 | 32.62 | 266.7
pedestrian-detection-adas-0002 | - | 4.7 | 28.23 | 267.0

### Using CPU Intel Core i7-6700K:
Model | Size before conversion [MB] | Size after conversion [MB] | Average inference Speed [ms] | Memory Usage [MB]
------|----------------------------|-----------------------------|------------------------------|------------------
ssd_mobilenet_v1_coco | 59.5 | 27.3 | 35.63 | 114.3 
ssd_resnet50_v1 | 270.2 | 206.9 | 441.32 | 554.2
ssd_mobilenet_v2_coco | 140.8 | 67.4 | 28.26 | 198.4
pedestrian-detection-adas-0002 | - | 4.7 | 24.3 | 82.7

All models have been tested with probability thresholds of: 0.2; 0.3; 0.5; 0.8 and 0.95.

It is observed that the most efficient model ends up being the "Pedestrian-detection-adas-0002" embedded within the algorithm that includes the OpenVino v2019.R3 toolkit.


## Assess Model Use Cases

In my case, because I work in industrial environments, I find it very useful to use this type of tools in personal safety systems, such as: detection of personnel working in areas with movements of systems that move loads (cranes, for instance) on top of them; in this case, the advance of the same towards where the personnel are could be disabled by the automation system, preventing incidents and/or accidents.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

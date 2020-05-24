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









My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

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

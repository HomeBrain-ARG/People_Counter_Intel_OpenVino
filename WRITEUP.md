# Intel® Edge AI for IoT Developers Nanodegree Program

# Project N°1 - Write-Up:


To develop the present project, tests were carried out with three pre-trained TensorFlow models and the results obtained were compared against a pre-trained Intel® model. In summary, I can said that much better results were obtained with the Intel® pre-trained model, I tested the models on CPU and GPU.


## Explaining Custom Layers:

En nuestro proyecto se han utilizado modelos pre-entrenados de TensorFlow, según hemos observado, cuanto más complejos han sido los modelos escogidos, mayor cantidad de custom layers hemos encontrado. Tener en cuenta que nuestra aplicación detecta si hay layers no soportados y menciona cuales son.

In our project, pre-trained TensorFlow models have been used, as we have observed, the more complex the chosen models have been, the more custom layers we have found.

Debido a la complejidad del proceso, he 
seguido minuciosamente los pasos indicados en la documentación de OpenVino v2019.R3: 

https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Offloading_Sub_Graph_Inference.html

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...


## Comparing Model Performance


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

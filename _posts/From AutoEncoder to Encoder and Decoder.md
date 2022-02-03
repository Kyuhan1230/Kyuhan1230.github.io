---
layout: single
title:  "Autoencoder to encoder/decoder"
categories: "ML/DL"
tags: [Python, MachineLearning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
# search: false
---

# ML/DL - AutoEncoder

```python
import warnings
warnings.filterwarnings(action='ignore') #default로 바꾸면 warning이 보임.

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
```

## AutoEncoder 모델 로드


```python
autoencoder= load_model('C:/Users/Projects/Model/MODEL.h5')
autoencoder.summary()
```

    WARNING:tensorflow:From C:\Users\asdm1\Anaconda3\envs\techdas\lib\site-packages\tensorflow\python\ops\init_ops.py:97: calling GlorotUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    WARNING:tensorflow:From C:\Users\asdm1\Anaconda3\envs\techdas\lib\site-packages\tensorflow\python\ops\init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    WARNING:tensorflow:From C:\Users\asdm1\Anaconda3\envs\techdas\lib\site-packages\tensorflow\python\ops\init_ops.py:97: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    Model: "AutoEncoder"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    InputLayer_MM (InputLayer)   [(None, 40)]              0         
    _________________________________________________________________
    MappingLayer_MM (Dense)      (None, 80)                3280      
    _________________________________________________________________
    LatentLayer_MM (Dense)       (None, 20)                1620      
    _________________________________________________________________
    DecodingLayer_MM (Dense)     (None, 80)                1680      
    _________________________________________________________________
    OutputLayer_MM (Dense)       (None, 40)                3240      
    =================================================================
    Total params: 9,820
    Trainable params: 9,820
    Non-trainable params: 0
    _________________________________________________________________


## AutoEncoder --> Encoder


```python
# create the encoder model
encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
encoder.summary()
encoding_dim = int(encoder.output.shape[1])
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    InputLayer_MM (InputLayer)   [(None, 40)]              0         
    _________________________________________________________________
    MappingLayer_MM (Dense)      (None, 80)                3280      
    _________________________________________________________________
    LatentLayer_MM (Dense)       (None, 20)                1620      
    =================================================================
    Total params: 4,900
    Trainable params: 4,900
    Non-trainable params: 0
    _________________________________________________________________


## AutoEncoder --> Decoder


```python
# create the decoder model
decoded_input = Input(encoding_dim,)
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
decoder = Model(decoded_input, decoder_layer2(decoder_layer1(decoded_input)))
decoder.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 20)]              0         
    _________________________________________________________________
    DecodingLayer_MM (Dense)     (None, 80)                1680      
    _________________________________________________________________
    OutputLayer_MM (Dense)       (None, 40)                3240      
    =================================================================
    Total params: 4,920
    Trainable params: 4,920
    Non-trainable params: 0
    _________________________________________________________________


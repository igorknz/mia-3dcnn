## MIA-3DCNN
This repository contains the scripts used to generate the results that our team submitted to the *IEEE ICASSP 2023: 3rd COV19D Competition*, for the detection challenge.

## Our team
 - Igor Kenzo Ishikawa Oshiro Nakashima, School of Electrical and Computer Engineering, Unicamp, Brazil;
 - Giovanna Vendramini, Institute of Computing, Unicamp, Brazil;
 - Hélio Pedrini, Ph.D., Institute of Computing, Unicamp, Brazil.

## Dataset
The dataset provided for training and validation was divided according to the table below.

| Set       | Training | Validation |
| :-------: | :------: | :--------: |
| COVID     | 922      | 225        |
| Non-COVID | 2110     | 489        |

Each sample corresponded to a directory containing slices of a CT scan from a patient. Each one of these folders contained a varying number of images, ranging from 50 to 700.

## Data processing
Due to the size of dataset that had to be used and our hardware limitations, the images had to be processed before training. We used a spline interpolation to reduce all images to 224x224 pixels, and to make all sample folders have 64 images.

## Our method
The MIA-3DCNN network is a 3D convolutional neural network, which is the method chosen to be used in the detection task. Due to the limited amount of samples, we used data augmentation to have more (and more diverse) samples.

### Convolutional neural network
Firstly, the neural network contains a number of stacked blocks composed of 3D convolutional layers, followed by a 3D max pooling layer, a batch normalization layer, and a dropout layer. After these initial blocks, we stacked a 3D global average pooling layer, followed by a couple of dense layers, each of them followed by a dropout layer. At last, there is a 2-neuron layer, for the classification.

### Data augmentation
The operations of data augmentation used were: additive Gaussian noise, Gaussian blur, rotation, flip (vertical and horizontal), cutout and gamma contrast.
 
### Versions
We provided two results to the competition, one from a model trained with data augmentation (version A) and another model trained without data augmentation (version B). In both versions, the model architecture is the same.
 
## Code usage
1. The data provided for the challenge should already have been downloaded, unzipped, and organized into the correct folders.
 
 ```
 train
    └─ covid
       └─ folder_0
           └─ 0000.png
           └─ 0001.png
              ...
           └─ 0063.png
          ...
          
    └─ non-covid
       └─ folder_0
          ...
     
  val
    └─ covid
       └─ folder_0
          ...
          
    └─ non-covid
       └─ folder_0
          ...
 ```
 
2. Run <code>preprocess_images.py</code>. Initially, all the original images will be renamed to xxxx.png, where xxxx is the number of the original image, preceded by zeros, so that the name of the image is formed by 4 algarisms. Then, all the images are resized to 224x224 pixels, and all samples folders will have 64 images. 

3. Run <code>train_mia_9a.py</code> to run the whole training pipeline. Make sure to supply the correct inputs to the parser.

4. The weights will be saved in the folder path supplied to the parser of the <code>train_mia_9a.py</code> script. With this, run the <code>valid_mia_9a.py</code> to run the inference in the test data. For this, the test data provided will have to be already downloaded and unzipped.

## Acknowledgments
We thank:
 - CENAPAD for providing the hardware where all the tests were run;
 - Semantix Brasil for financing this project.

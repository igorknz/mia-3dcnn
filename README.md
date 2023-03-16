## MIA-3DCNN
This repository contains the scripts used to generate the results that our group submitted to the *IEEE ICASSP 2023: 3rd COV19D Competition*, for the detection challenge. In this task, we had to create a method that classifies each sample as covid or non-covid.

## Dataset
The dataset provided for training and validation was divided according to the table below.

| Set       | Training | Validation |
| :-------: | :------: | :--------: |
| Covid     | 922      | 225        |
| Non-covid | 2110     | 489        |

Each sample corresponded to a directory containing slices of a CT scan from a patient. Each one of these folders contained a varying number of images, ranging from 50 to 700.

## Data processing
Due to the amount of data that hat to be used and our hardware limitations, the images had to be processed before training. We used 


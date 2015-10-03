Kaggle Denoising Dirty Documents
====

## Model

Deep Convolutional Network imspired by SRCNN and NIN

|Layer Type|#Input Channel|#Ouput Channel|FilterSize|Padding|
|:--:|:--:|:--:|:--:|:--:|
|Input| - |1|-|-|
|Convolution(1)|1|96|3|1|
|Convolution(2)|96|96|1|1|
|Convolution(3)|96|96|3|1|
|Convolution(4)|96|96|1|1|
|Convolution(5)|96|96|3|1|
|Convolution(6)|96|1|3|1|
|Output|96|1|-|-|

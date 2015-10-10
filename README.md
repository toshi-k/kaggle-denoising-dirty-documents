Kaggle Denoising Dirty Documents
====

My solution in this Kaggle competition ["Denoising Dirty Documents"](https://www.kaggle.com/c/denoising-dirty-documents) (6th place).


## Model

Deep Convolutional Neural Network

|Layer Type|#Input Channel|#Ouput Channel|FilterSize|Padding|
|:--:|:--:|:--:|:--:|:--:|
|Input| - |1|-|-|
|Convolution(1)|1|96|3|1|
|Leaky ReLU|96|96|-|-|
|Convolution(2)|96|96|1|0|
|Leaky ReLU|96|96|-|-|
|Convolution(3)|96|96|3|1|
|Leaky ReLU|96|96|-|-|
|Convolution(4)|96|96|1|0|
|Leaky ReLU|96|96|-|-|
|Convolution(5)|96|96|3|1|
|Leaky ReLU|96|96|-|-|
|Convolution(6)|96|1|3|1|
|Output|1|-|-|-|

## Sample

Test Input<br>
![sample_test_input](https://raw.githubusercontent.com/toshi-k/Kaggle-Denoising-Dirty-Documents/master/sample/test_input.png)

Test Output<br>
![sample_test_output](https://raw.githubusercontent.com/toshi-k/Kaggle-Denoising-Dirty-Documents/master/sample/test_output.png)
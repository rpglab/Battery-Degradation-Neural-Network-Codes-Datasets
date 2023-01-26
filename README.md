## Battery Degradation Neural Network

This resource page includes the dataset used to train a battery degradation neural network model, as well as the associated python code.


### File Description
* 'DNN.py' is designed to train a deep neural network model to predict the battery degradation value.
* 'BatteryData_Sample.xlsx' is a sample battery data that collected processed from matlab. which is also the input for the 'readtestdata_size.py'
* 'readtestdata_size.py' is designed to process the data from the excel and pack the data into x_sample,y_sample,x_test,y_test.
* 'x_sample.np ,y_sample.np ,x_test.np ,y_test.np' are the tranining datas already attached. Normally it is the output of 'readtestdata_size.py'


### Environment (Python packages)
* torch
* numpy
* matplotlib
* os

Note the if you have a cuda supported GPU, we highly suggest you to install the cuda package to speed the training process.  https://pytorch.org/get-started/locally/

In our case, the training time is reduced by six times than non-cuda involved traning. 

The code is comaptiable with both GPU training and CPU traing (non-cuda).


### Input Parameters:
* file = 'xxxxxxxxx.xlsx'
* sample_num = how many battery aging tests for training dataset.
* cycle_num = how many cycles for each aging tests.
* feature_num = number of input features
* test_num = how many battery aging tests for testing dataset
* seq_length = cycle_num  
* BATCH_SIZE = trainig batch size
* epoches = traning epoches
* training_num = total number of traning data cell
* validation_num = total number of validation data cell



## Citation:
If you use these codes for your work, please cite the following paper:

Cunzhi Zhao and Xingpeng Li, “Microgrid Optimal Energy Scheduling Considering Neural Network based Battery Degradation”, *IEEE Transactions on Power Systems*, early access, Jan. 2023.


Paper website: <a class="off" href="/papers/CunzhiZhao-NNBD-MDS/"  target="_blank">https://rpglab.github.io/papers/CunzhiZhao-NNBD-MDS/</a>


## Contributions:
Cunzhi Zhao developed this program. Xingpeng Li supervised this work.


## Contact:
If you need any techinical support, please feel free to reach out to Cunzhi Zhao at czhao20@uh.edu.

For collaboration, please contact Dr. Xingpeng Li at xli83@central.uh.edu.

Website: https://rpglab.github.io/


## License:
This work is licensed under the terms of the <a class="off" href="https://creativecommons.org/licenses/by/4.0/"  target="_blank">Creative Commons Attribution 4.0 (CC BY 4.0) license.</a>


## Disclaimer:
The author doesn’t make any warranty for the accuracy, completeness, or usefulness of any information disclosed and the author assumes no liability or responsibility for any errors or omissions for the information (data/code/results etc) disclosed.

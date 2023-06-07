# Covid19 New Cases Analysis LSTM

Creating a model prediction for Covid19 new cases in Malaysia using deep learning approach with LSTM neural network

### Description
Objective: Create a deep learning model using LSTM neural
network to predict new cases in Malaysia using the past 30 days
of number of cases

* Model training - Deep learning
* Method: Sequential, LSTM
* Module: Sklearn & Tensorflow

In this analysis, dataset used from https://github.com/MoH-Malaysia/covid19-public

### About The Dataset:
There are 2 dataset used in this analysis:-
1. cases_malaysia_train.csv (680 data entries with 31 column)
   25/1/2020-4/12/2021
2. cases_malaysia_test.csv (100 data entries with 31 column)
   5/12/2021-14/3/2022

To predict new cases, we only focus on 'cases_new' column. There are few missing data and symbol found and data cleaning process were applied.

### Deep learning model with LSTM layer
A sequential model was created with 3 LSTM layer, 3 Dropout layer, 1 Dense layer:
<p align="center">
  <img src="https://github.com/Ghost0705/Covid19-New-Cases-Analysis-LSTM/blob/main/picture/model_architecture.PNG">
</p>

<p align="center">
  <img src="https://github.com/Ghost0705/Covid19-New-Cases-Analysis-LSTM/blob/main/picture/model_flow.PNG">
</p>

Data were trained with 800 epoch:
<p align="center">
  <img src="https://github.com/Ghost0705/Covid19-New-Cases-Analysis-LSTM/blob/main/picture/model_performance.PNG">
</p>

### Result
<p align="center">
  <img src="https://github.com/snaffisah/Covid19-New-Cases-Analysis-LSTM/blob/main/picture/actual_vs_predict.PNG">
</p>

After the deployment of model mean absolute percentage error able to achieve 0.1% and the model is good enough to be used for future new cases prediction and goverment can take the necessary precaution steps to avoid it from spreading.

### How to run the pythons file:
Run training file 'Covid-19_prediction.py' 

Enjoy!


# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the [Udacity Azure ML Nanodegree](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333).

* In this project, an [Azure](https://azure.microsoft.com/en-us/) ML pipeline was built and optimized using the [Azure Python SDK](https://docs.microsoft.com/en-us/azure/developer/python/azure-sdk-overview) and a provided [Scikit-learn](https://scikit-learn.org/stable/) model.
* The Scikit-learn model used was a [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier.
* Hyperparameter optimization of this model was carried out using [Azure HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py). 
* This model was then compared to the results obtained using an [Azure AutoML](https://azure.microsoft.com/en-us/services/machine-learning/automatedml/) run.

The main steps required  to create, optimize and compare the output of a Scikit-learn ML-Pipeline against AutoML are illustrated below (figure sourced from [Udacity Azure ML Nanodegree](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333) course notes): 

![alt text](./img/creating-and-optimizing-an-ml-pipeline.png "Creating and optimizing an ml-pipeline")


## Summary
### Problem Statement

**Dataset**
* The dataset is derived from the direct marketing campaigns of a Portuguese banking institution and represents a binary classification task.
* The target classification denotes whether a given financial product: _bank term deposit_, would be subscribed to or not, denoted **yes** and **no** respectively.

**Aim**
* The aim is to train models which can learn to predict the target class by fitting the input features in the dataset using machine learning algorithms.

The dataset used in this project was sourced [here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv).

Some inspection of the dataset can be viewed [here](./Inspect_Data/).



### Outcome
**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline

### The pipeline architecture: 

#### Data Preparation
* **Data acquisition**: the raw data, sourced [here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv), was read into a [TabularDataset](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabular_dataset.tabulardataset) using a [TabularDatasetFactory](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory) class as shown in the following code:
```  
raw_data_url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(raw_data_url)
```
* **Data cleaning**; a function was created to handle data cleaning in which the following steps were carried out:
  - Rows containing missing values were removed (for this dataset no rows needed to be removed).
  - Certain categorical features were converted to a One-Hot-Encoding in order to make the features consumable for ML models and preserve the categorical information. 
  - Conversion of some categorical features to binary, e.g. the feature `marital`, which indicates marital status, was converted from a 4 category feature to a binary feature indicating **married** or **not-married**.
  - Categorical features `month` and `day of week` were mapped to an integer encoding which preserves the relative temporal information.
    
* **Create train and test set**:
    - Scikit-learn's `train_test_split` [function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) was used to create a train and test set in which 33% of the available data was assigned to the test set. 
    - The `train_test_split` function's `stratify` parameter was used to ensure that the distribution of the target class in the full dataset would the same in the train and test sets (this was achieved by passing the target data column to the stratify parameter).
    - The data is ver imbalanced. The use of the `stratify` parameter...
    
#### Classification Algorithm.
* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier.
* Which parameters were investigated.
* Performance metric - accuracy.

#### Hyperparameter Tuning
* HyperDrive

**What are the benefits of the parameter sampler you chose?**

The [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling) class was used.
```
param_dict = {
    "--C" : uniform(0.001, 2.0),
    "--max_iter" : choice(10,25,50,100,150,200,250)
}
ps = RandomParameterSampling(param_dict)
```

**What are the benefits of the early stopping policy you chose?**

The [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy) was used.
```
policy = BanditPolicy(evaluation_interval=5, slack_factor=0.2)
```

#### Saving the best model

* Download (to local outputs directory)
* Register

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance.** 
* What are the differences in accuracy? 
* In architecture? 
* If there was a difference, why do you think there was one?

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
* Use AUC as criterion due to the dataset being imbalanced.
* Make use of the Logistic Regression Classifier's weighting parameter.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

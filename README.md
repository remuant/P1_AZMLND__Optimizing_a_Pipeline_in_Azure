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

Some inspection of the dataset and additional information about the data can be viewed [here](./Inspect_Data/) where a Jupyter notebook can be found containing implementation which allows the target distribution for different features to be viewed.



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
* **Data cleaning**; a function, `clean_data`, was created to handle data cleaning in which the following steps were carried out:
  - Rows containing missing values can be removed (for this particular dataset no rows needed to be removed).
  - Certain categorical features were converted to a One-Hot-Encoding in order to make the features consumable for ML models and preserve the categorical information. 
  - Conversion of some categorical features to binary, e.g. the feature `marital`, which indicates marital status, was converted from a 4 category feature to a binary feature indicating **married** or **not-married**.
  - Categorical features `month` and `day of week` were mapped to an integer encoding which preserves the relative temporal positioning information.
    
* **Create train and test sets**:
    - Scikit-learn's `train_test_split` [function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) was used to create a train and test set in which 33% of the available data was assigned to the test set. 
    - The `train_test_split` function's `stratify` parameter was used to ensure that the distribution of the target class in the train and test sets would the same as that found in the full dataset (this was achieved by passing the target data column to the stratify parameter).
    - **Note**: The data is very imbalanced with 88% of the examples being in the **_no_** target category (see [here](./Inspect_Data/) for some inspection of the data). The use of the `stratify` parameter was implemented to ensure that the imbalance of the target distribution was not made more extreme in either the training or test sets after random sampling.
    
#### Classification Algorithm.
* The [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier was used in the Scikit-learn pipeline.
* Logistic Regression uses a linear regression equation to produce discrete binary outputs. 
* The parameters which were investigated:
  - `C`: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
  - `max_iter`: Maximum number of iterations taken for the solvers to converge.
  - **Note**: Given the imbalanced distribution of the target values in the dataset, it would also be appropriate to investigate parameter settings for the `class_weight` parameter, see **Future Work** section.
    - Weights associated with classes in the form `{class_label: weight}`. If not given, all classes are supposed to have weight one.

* Performance metric:
  - **Test set accuracy**, i.e. the total sum of the **True Postive** and **True Negative** classifications as a percentage of the total available test examples.
  - **Note**: Area Under Curve (AUC) may have been a more appropriate choice due to dataset imbalance, see **Future Work** section.

#### Hyperparameter Tuning
Hyperparameter optimization of the [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier was carried out using [Azure HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py).

##### Parameter Sampler:
This required the creation of a [parameter sampler](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) to select and combine the parameter values to be optimized.

Three choices of [parameter sampler](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) were available:
* [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling)
  - In random sampling, hyperparameter values are randomly selected from the defined parameter-value search space.
  - Discrete and continuous choice of hyperparameters are supported.
  - Early termination of low-performance runs is supported.
* [GridParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.gridparametersampling)
  - In grid sampling an exhaustive search of the defined parameter-value search space is carried out.
  - **Only discrete choice** of parameter-values is supported.
  - Early termination of low-performance runs is supported.
* [BayesianParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.bayesianparametersampling)
  - Based on the Bayesian optimization algorithm.
  - Bayesian parameter sampling selects samples based on how previous samples performed, so that new samples improve the primary metric.
  - Bayesian sampling only supports discrete choice, uniform, and quniform distributions over the parameter-value search space.
  - Early termination of low-performance runs is **NOT supported**.

[RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling) was selected as it is the least computationally expensive of the three choices and supports discrete and continuous selection of parameter-values, i.e. a set of discrete parameter-values can be defined as well as a range over which parameter-values should be selected could be defined.
Support for early termination of low-performance runs also made this sampler an attractive choice. This is an efficient strategy for at least identifying a promising area of the hyperparameter space before applying the more computationally expensive [Grid](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.gridparametersampling) and [Bayesian](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.bayesianparametersampling) sampling strategies. 

Using [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling):
* A **parameter dictionary** was defined which sets out which parameters are to be optimized and over which range or set of values. 
* The code snippet below shows the parameter dictionary used optimized the `C` parameter over a **uniform** distribution in the range specified, and the `max_iter` parameter over a discrete **choice** of specified values.
* The parameter dictionary was then passed to the [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling) class at initialization.
```
param_dict = {
    "--C" : uniform(0.001, 2.0),
    "--max_iter" : choice(10,25,50,100,150,200,250)
}
ps = RandomParameterSampling(param_dict)
```

##### Early Termination Policy:
**What are the benefits of the early stopping policy you chose?**

Four choices of [early termination policy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) were available:
* [Bandit policy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#bandit-policy)
  - This policy is based on slack factor/slack amount and evaluation interval.
  - Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.
* [Median stopping policy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#median-stopping-policy)
  - This is an early termination policy based on running averages of primary metrics reported by the runs.
  - This policy computes running averages across all training runs and terminates runs with primary metric values worse than the median of averages.
* [Truncation selection policy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#truncation-selection-policy)
  - This policy cancels a percentage of the lowest performing runs at each evaluation interval.
  - Runs are compared using the primary metric.
* [No termination policy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#no-termination-policy-default)
  - If no policy is specified, the hyperparameter tuning service will let all training runs execute to completion.



The [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy) was used.
It was chosen because it allows an aggressive policy for termination of runs to be enabled by using criteria which ensure that surviving runs are within a margin of the best run defined by the slack factor.
The criteria used by the MedianStoppingPolicy is less aggressive as it considers a criterion which is based on an aggregate of the performance across runs. The TruncationSelectionPolicy terminates a percentage of the lowest performing runs which could leave other poor runs to continue.

The [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy) was implemeted as shown in this code snippet:
```
policy = BanditPolicy(evaluation_interval=5, slack_factor=0.2)
```

#### Saving the best model

* First the best run needs to be fetched using `get_best_run_by_primary_metric`:

```
best_run = hd_run.get_best_run_by_primary_metric()
```
* Next, download the model to the local file system from the best run using the run object’s `download_file` method. 
  - Note: The model will be the last file in the list hence the -1 index can be used to reference it.
```
os.makedirs("outputs", exist_ok=True)  # Ensure that there is a local outputs folder
best_run.download_file(best_run.get_file_names()[-1], output_file_path='./outputs/')  # Download
```
+ Verify that best model has been retrieved:
```
joblib.load('./outputs/my_model.joblib')
```

* Register the model:
```
best_hyperdrive_model = best_run.register_model(
    model_name="best_hyperdrive_model",
    model_path="./outputs/my_model.joblib",
    tags=best_run.get_metrics()
)
```

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

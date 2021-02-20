from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# PLEASE NOTE that the implementation required for this assignment is located in main():
# 1. This ensures that the function call to clean_data is made after the function has been defined rather than before.
# 2. As the parameters originally defined as globals are required only inside of main(), the implementation is neater if
# they are declared there.

# Additional note: the return statement was originally missing from the clean_data function, this has been added.


def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()  # Remove rows with missing values
    # Useful reference:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
    # Convert categorical variable into dummy/indicator variables (OHE).
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    # Binary features:
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    # Convert categorical variable into dummy/indicator variables (OHE).
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    # Convert categorical variable into dummy/indicator variables (OHE).
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    # Map raw feature values for month and day to integers specified in dictionaries (this method ensures that relative
    # temporal position information is preserved).
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    # Binary feature:
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    # Extract the binary target column and convert to integer representation
    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)

    # Note: added missing return statement(!)
    # Return the input and target features
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # 1. Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    # Useful reference:
    # https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory
    raw_data_url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    ds = TabularDatasetFactory.from_delimited_files(raw_data_url)
    x, y = clean_data(ds)

    # 2. Split data into train and test sets.
    # Useful reference which explains how this works and can guide parameter choice:
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=8, stratify=y)
    # Note: Using the stratify parameter ensures that the split contains the same distribution of target values in the
    # training and test sets as the proportion of values in the entire dataset when the target data for the entire
    # dataset (in this case: y) is passed to the stratify parameter.

    # Useful reference:
    # https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # Note: It may also be worth investigating the 'class_weight' parameter in the LogisticRegression model to deal with
    # the dataset imbalance:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    # Note: could be worth trying out a different performance metric, e.g. AUC, due to dataset imbalance
    # (88% target outputs: 'no')
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    # Save the model
    # See these links for useful information:
    # https://knowledge.udacity.com/questions/424266
    # https://www.kaggle.com/pankaj1234/azure-machine-learning-model-training
    # https://towardsdatascience.com/azure-machine-learning-service-train-a-model-df72c6b5dc
    os.makedirs("outputs", exist_ok=True)  # Precautionary, creation should be automatic
    joblib.dump(value=model, filename="./outputs/my_model.joblib")


if __name__ == '__main__':
    main()

# library doc string
'''
Author: Thanh Luu
Project: Churn Dectection
'''

# import libraries
import os
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
# from sklearn.metrics import plot_roc_curve
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
EDA_FOLDER = './images/eda'
RESULTS_FOLDER = './images/results'
MODEL_FOLDER = './models'
CATEGORY_LIST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        dataframe = pd.read_csv(pth)
    except FileNotFoundError:
        logging.error("Can not find file")
    else:
        dataframe.head()
        return dataframe
    return None


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(f'{EDA_FOLDER}/churn_distribution.png')

    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(f'{EDA_FOLDER}/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{EDA_FOLDER}/marital_status_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{EDA_FOLDER}/total_transaction_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        data=dataframe.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(f'{EDA_FOLDER}/heatmap.png')


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response(optional): string of response name 
            [argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        lst = []
        groups = dataframe.groupby(category)[response].mean()
        for val in dataframe[category]:
            lst.append(groups.loc[val])

        dataframe[category + '_' + response] = lst

    return dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              df: pandas dataframe
              response(optional): string of response name 
              [argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x = pd.DataFrame()
    y = dataframe[response]
    x[keep_cols] = dataframe[keep_cols]
    x.head()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f'{RESULTS_FOLDER}/rf_result.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f'{RESULTS_FOLDER}/lr_result.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='newton-cg', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    joblib.dump(cv_rfc.best_estimator_, f'{MODEL_FOLDER}/rfc_model.pkl')
    joblib.dump(lrc, f'{MODEL_FOLDER}/logistic_model.pkl')

    # rfc_model = joblib.load(f'{MODEL_FOLDER}/rfc_model.pkl')
    # lr_model = joblib.load(f'{MODEL_FOLDER}/logistic_model.pkl')

    feature_importance_plot(
        cv_rfc,
        x_train,
        f'{RESULTS_FOLDER}/feature_importance.png')
    

if __name__ == "__main__":
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    df.head()

    df = encoder_helper(dataframe=df, category_lst=CATEGORY_LIST, response='Churn')

    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe=df, response='Churn')
    train_models(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    # print(X_train + " " + X_test + " " + y_train + " " + y_test)

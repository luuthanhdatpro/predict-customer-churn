'''
Author: Thanh Luu
'''
import os
import logging
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

EDA_FOLDER = './images/eda'
RESULTS_FOLDER = './images/results'
MODEL_FOLDER = './models'
DATA_PATH = './data/bank_data.csv'


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df_test = cls.import_data(DATA_PATH)
    try:
        perform_eda(df_test)
        assert os.listdir(EDA_FOLDER)
        assert os.path.isfile(f"{EDA_FOLDER}/churn_distribution.png")
        assert os.path.isfile(f"{EDA_FOLDER}/customer_age_distribution.png")
        assert os.path.isfile(f"{EDA_FOLDER}/marital_status_distribution.png")
        assert os.path.isfile(
            f"{EDA_FOLDER}/total_transaction_distribution.png")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: The generated images don't exist")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df_test = cls.import_data(DATA_PATH)
    cls.perform_eda(df_test)
    try:
        encoder_helper(df_test, cls.CATEGORY_LIST, 'Churn')
        assert df_test.shape[0] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError:
        logging.error("Testing encoder_helper: FAILED")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_test = cls.import_data(DATA_PATH)
    cls.perform_eda(df_test)
    cls.encoder_helper(df_test, cls.CATEGORY_LIST, 'Churn')
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe=df_test, response='Churn')
        assert x_train.shape[0] > 0 and x_test.shape[0] > 0 \
            and y_train.shape[0] > 0 and y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df_test = cls.import_data(DATA_PATH)
    cls.perform_eda(df_test)
    cls.encoder_helper(df_test, cls.CATEGORY_LIST, 'Churn')
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        dataframe=df_test, response='Churn')
    try:
        train_models(x_train, x_test, y_train, y_test)
        assert os.listdir(MODEL_FOLDER)
        assert os.listdir(RESULTS_FOLDER)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The generated images don't exist")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

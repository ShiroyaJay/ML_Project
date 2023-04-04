import os
import sys
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()
    
    def get_data_transformaer_object(self):
        try:
            numerical_columns=["writing_score", "reading_score"]
            categorical_columns=[ "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            

            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

            ])
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scale",StandardScaler(with_mean=False))

            ])

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")


            preprocessing_obj=self.get_data_transformaer_object()


            target_column_name='math_score'

            x_train=train_df.drop([target_column_name],axis=1)
            y_train=train_df[target_column_name]

            x_test=test_df.drop([target_column_name],axis=1)
            y_test=test_df[target_column_name]


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            x_train_after_fit_tran_arr=preprocessing_obj.fit_transform(x_train)
            x_test_fit_arr=preprocessing_obj.transform(x_test)

            train_arr=np.c_[
                x_train_after_fit_tran_arr,np.array(y_train)
            ]
            test_arr=np.c_[
                x_test_fit_arr,np.array(y_test)
            ]
            logging.info(f"Saved preprocessing object.")

            save_object(file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)


            return(
               train_arr, test_arr,
               self.data_tranformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
                
            
           




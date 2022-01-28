import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *

# Split the dataset for test purpose and
# a large part for training.
X = dataset.drop('Buyer',axis=1)
y = dataset[['Buyer']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05)


# Merge the feature training dataset and target training column in a unique training dataset 
df_training = pd.concat([X_train, y_train], axis=1, ignore_index=True).reset_index(drop=True, inplace=False)
# Create an array concatenating column names of features and target variable
col_names = np.concatenate((X.columns, y.columns),axis=0)
# Assign column names to the new dataframe
df_training = df_training.set_axis(col_names,axis=1)

# Force the float values of the Occupation attribute to an int datatype
df_training['Occupation'] = df_training['Occupation'].astype('int')

# Setup the ML parameters 
exp_clf = setup(data = df_training, target = 'Buyer', session_id=5614,
                categorical_features=['Age', 'MaritalStatus', 'NumberCarsOwned'],
                ordinal_features={'Occupation' : [0,1,2,3,4]},
                n_jobs=1, # remove parallelism to work in Power BI
                silent=True,
                verbose=False)

# Get the best performing model
best_model = compare_models(verbose=True)


# Save the model in a pkl file for future reuse
save_model(best_model, r'C:\Users\MrLeo\sales-model')

# Merge the feature test dataframe and target test column in a unique test dataframe
# and save it in a CSV file for future reuse
df_test = pd.concat([X_test, y_test], axis=1, ignore_index=True).reset_index(drop=True, inplace=False)
df_test = df_training.set_axis(col_names,axis=1)
df_test.to_csv(r'C:\Users\MrLeo\sales-model.csv',
               index=False)

# Get model predictions for the test dataframe
predictions = predict_model(best_model,data = df_test,verbose=False)
predictions

import numpy as np 
import pandas as pd 

from sklearn.impute import KNNImputer 
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from pycaret.classification import *

# Get the column indexes of categorical features 
#modeldir = "C:\Users\MrLeo\sales-model"
trainurl = "https://raw.githubusercontent.com/LeonOnyx/Onyx_PowerBI_ML/main/SalesTraining.csv"
leadurl = "https://raw.githubusercontent.com/LeonOnyx/Onyx_PowerBI_ML/main/ProspectiveBuyer.csv"
train_dataset = pd.read_csv(trainurl)
lead_dataset = pd.read_csv(leadurl)
categorical_idx = train_dataset.select_dtypes(include=['object', 'bool']).columns 

# Define the transformer for categorical features. 
# A transformer is a three-element tuple defined by the name of the transformer, the transform to apply, 
# and the column indices to apply it to. In this case we apply the ordinal encoding to each categorical column 

t = [('cat', OrdinalEncoder(), categorical_idx)] 
col_transform = ColumnTransformer(transformers=t) 

 
# Get the transformed columns 
X = col_transform.fit_transform(train_dataset) 


# Replace the categorical columns of a new dataframe with the transformed columns 
df_transf = train_dataset.copy() 
df_transf[categorical_idx.tolist()] = X 


# Let's impute Age using knn algorithm 
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean',  
                     missing_values=np.nan, add_indicator=False) 


# Fit on cleaned dataframe 
imputer.fit(df_transf) 

# Transform the dataframe according to the imputer and get the imputed matrix 
matrix_imputed = imputer.transform(df_transf) 

# Let's transform back the matrix into a dataframe using the same 
# column names of the cleaned dataframe 

df_imputed = pd.DataFrame(matrix_imputed, columns=train_dataset.columns) 
df_imputed['Buyer'] = df_imputed['Buyer'].astype('int') 

# Split the dataset for test purpose and
# a large part for training.
X = df_imputed.drop('Buyer',axis=1)
y = df_imputed[['Buyer']]

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

categorical_idx = lead_dataset.select_dtypes(include=['object', 'bool']).columns 

# Define the transformer for categorical features. 
# A transformer is a three-element tuple defined by the name of the transformer, the transform to apply, 
# and the column indices to apply it to. In this case we apply the ordinal encoding to each categorical column 

t = [('cat', OrdinalEncoder(), categorical_idx)] 
col_transform = ColumnTransformer(transformers=t) 

 
# Get the transformed columns 
X = col_transform.fit_transform(lead_dataset) 


# Replace the categorical columns of a new dataframe with the transformed columns 
df_transf = lead_dataset.copy() 
df_transf[categorical_idx.tolist()] = X 


# Let's impute Age using knn algorithm 
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean',  
                     missing_values=np.nan, add_indicator=False) 


# Fit on cleaned dataframe 
imputer.fit(df_transf) 

# Transform the dataframe according to the imputer and get the imputed matrix 
matrix_imputed = imputer.transform(df_transf) 

# Let's transform back the matrix into a dataframe using the same 
# column names of the cleaned dataframe 

df_leadimputed = pd.DataFrame(matrix_imputed, columns=lead_dataset.columns) 

directory = r'C:\Users\MrLeo\sales-model'
rforest_model = load_model(directory)

# Get model predictions for the input dataset
predictions = predict_model(rforest_model,data = df_leadimputed,verbose=True)
#print(predictions.columns)
#print(lead_dataset.columns)

final_predictions = pd.merge(predictions[['ProspectiveBuyerKey','Label','Score']], 
                lead_dataset[['ProspectiveBuyerKey','FirstName','LastName']], 
                on = "ProspectiveBuyerKey", how = "inner")

#Clean up dataframes
del df_imputed
del df_leadimputed
del df_test
del df_training
del df_transf
del train_dataset
del X_test
del X_train
del y
del y_test
del y_train
del lead_dataset
del predictions

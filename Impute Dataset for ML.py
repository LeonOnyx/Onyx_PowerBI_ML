import numpy as np 
import pandas as pd 

from sklearn.impute import KNNImputer 
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.compose import ColumnTransformer 
  
 

# %% 

# Get the column indexes of categorical features (Sex and Embarked) 

categorical_idx = dataset.select_dtypes(include=['object', 'bool']).columns 

# Let's define the transformer for categorical features. 

# A transformer is a three-element tuple defined by the name of the transformer, the transform to apply, 

# and the column indices to apply it to. In this case we apply the ordinal encoding to each categorical column 

t = [('cat', OrdinalEncoder(), categorical_idx)] 

col_transform = ColumnTransformer(transformers=t) 

  

# Get the only two transformed columns 

X = col_transform.fit_transform(dataset) 

  

# Replace the Sex and Embarked columns of a new dataframe with the transformed columns 

df_transf = dataset.copy() 

df_transf[categorical_idx.tolist()] = X 

  

# %% 

# Let's impute Age using knn algorithm 

imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean',  

                     missing_values=np.nan, add_indicator=False) 

  

# Fit on cleaned dataframe 

imputer.fit(df_transf) 

# Transform the dataframe according to the imputer and get the imputed matrix 

matrix_imputed = imputer.transform(df_transf) 

  

# %% 

# Let's transform back the matrix into a dataframe using the same 

# column names of the cleaned dataframe 

df_imputed = pd.DataFrame(matrix_imputed, columns=dataset.columns) 

df_imputed['Buyer'] = df_imputed['Buyer'].astype('int') 
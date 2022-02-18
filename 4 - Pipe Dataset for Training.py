from pycaret.classification import *
directory = r'C:\Users\MrLeo\sales-model'
rforest_model = load_model(directory)

# Get model predictions for the input dataset
predictions = predict_model(rforest_model,data = dataset,verbose=True)
predictions

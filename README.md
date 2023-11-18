conda create -p venv python==3.8 -y
conda activate venv/ 
pip install -r requirements.txt

data import:
define paths for raw train test data to save
read data
make directories
split to train and test
save csvs
return train and test paths for next steps (transformation, model)

data transform:
define path for preprocessor object
functions: create transformer and initiate transformer on data
transformer: Pipeline - Imputers - Column Transformer (Preprocessor), return preprocessor
initiate : load data, load transformer, apply transformer to data, save transformer and transformed data, return train and test data in function for next steps (to model)

data_import
transform_data
model_trainer1
main
model_evaluation
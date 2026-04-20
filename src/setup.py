#Download dataset from kaggle

import kagglehub

path = kagglehub.dataset_download("excel4soccer/espn-soccer-data")

print("Path to dataset files:", path)
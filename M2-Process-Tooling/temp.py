import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

# Convert to DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Save to CSV
iris_df.to_csv('iris.csv', index=False)

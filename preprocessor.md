# Sklearn Preprocessing Tips

+ Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(3), LinearRegression())
```

+ Preprocessor

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

num_features = [3,4]
cat_features = [0,1,2,5]
numeric_transformer = make_pipeline(StandardScaler())
category_transformer = make_pipeline(OneHotEncoder())

preprocessor = ColumnTransformer(transformers=[
  ('num', numeric_transformer, num_features),
  ('cat', category_transformer, cat_features)
])
pipeline = make_pipeline(preprocessor, LogisticRegression(C=100, solver='liblinear'))
```


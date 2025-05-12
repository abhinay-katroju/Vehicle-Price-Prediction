import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df= pd.read_csv('/content/vehicle price prediction.csv')

df

df.isnull().sum()

df.duplicated().sum()

df = df.drop('name', axis=1)
df.head()

df.describe(include='all')

df_cat=df.select_dtypes(include='object')

for column in df_cat.columns:
  unique_count=df[column].nunique()
  print(f"column '{column}' has {unique_count} unique values")

from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()
df['description'] = label_encoder.fit_transform(df['description'])
df['make'] = label_encoder.fit_transform(df['make'])
df['model'] = label_encoder.fit_transform(df['model'])
df['engine'] = label_encoder.fit_transform(df['engine'])
df['fuel'] = label_encoder.fit_transform(df['fuel'])
df['transmission'] = label_encoder.fit_transform(df['transmission'])
df['trim'] = label_encoder.fit_transform(df['trim'])
df['body'] = label_encoder.fit_transform(df['body'])
df['exterior_color'] = label_encoder.fit_transform(df['exterior_color'])
df['interior_color'] = label_encoder.fit_transform(df['interior_color'])
df['drivetrain'] = label_encoder.fit_transform(df['drivetrain'])
df.head()

x = df.drop('price', axis=1)
y = df['price']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=365)

!pip install scikit-learn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer # Import SimpleImputer

imputer_x = SimpleImputer(strategy='mean') # Create an imputer for 'x' with 'mean' strategy
x_scaled = imputer_x.fit_transform(x) # Fit and transform 'x' to replace NaNs with the mean

# Handle missing values in 'y' by removing rows with NaN in 'price'
df = df.dropna(subset=['price'])  # Remove rows with NaN in 'price' column

# Redefine x and y after removing rows with NaN in 'price'
x = df.drop('price', axis=1)
y = df['price']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Fit the Linear Regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_train)

y_test_pred = reg.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_train, y_pred))
print(r2_score(y_test, y_test_pred))

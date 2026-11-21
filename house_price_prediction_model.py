
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---load dataset---
data = pd.read_csv(r'train.csv')
print(data.head())
print(data.info())
print(data.describe(include='all'))

# ---Data preprocessing---
# find and remove null values
print(data.isnull().sum())
print(data['LotFrontage'])


# ---convert string into numbers---
data['LotFrontage'] = pd.to_numeric(data['LotFrontage'], errors='coerce')
# fill null values through neighborhood median
data['LotFrontage'] = data.groupby("Neighborhood")['LotFrontage']\
                          .transform(lambda x: x.fillna(x.median()))
data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())
print(data['LotFrontage'].isna().sum())


# Text columns none: "None"
for c in data.select_dtypes(include="object").columns:
    data[c] = data[c].fillna("None")

# Number columns: median
for c in data.select_dtypes(include="number").columns:
    data[c] = data[c].fillna(data[c].median())

    
# ---  check missing values all columns ---
missing = data.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))


# ----handle duplicates values----
print(data.duplicated().sum())   #duplicates 0


# ----check data types----
print(data.dtypes)
print(data.dtypes.value_counts())
text_cols = data.select_dtypes(include="str").columns
print(list(text_cols))


# -----invalid values and outliers----
# find invalid values
numbers = data.select_dtypes(include='number')
print((numbers < 0).sum().sum())
print((data['YearBuilt'] > data['YrSold']).sum())


# find outliers
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.xlabel("GrlivArea")
plt.ylabel("SalePrice")
plt.show()
data = data[~((data['GrLivArea'] > 4000) & (data['SalePrice']< 300000))]
print(data.shape)


# -----feature engineering------
data['HouseAge'] = data['YrSold']-data['YearBuilt']
data['RemodAge'] = data['YrSold']-data['YearRemodAdd']
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBath'] = (data['FullBath'] + 0.5 * data['HalfBath']
                     + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath'])
print(data[["HouseAge", "RemodAge", "TotalSF", "TotalBath"]].describe())



# ------Encoding------
txt_col = data.select_dtypes(include="str").columns
print(len(txt_col))

grade = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

quality_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                "HeatingQC", "KitchenQual", "FireplaceQu",
                "GarageQual", "GarageCond", "PoolQC"]
for c in quality_cols:
    data[c] = data[c].fillna("None").map(grade)
print(data[quality_cols].isna().sum().sum())


data = pd.get_dummies(data, drop_first=True)
data = data.astype(float)
print(data.shape)
print(data.dtypes.value_counts())


# -----X, y and train_test split ---------
# input and output
X = data.drop(['SalePrice', 'Id'], axis=1)
y = np.log1p(data['SalePrice'])
print(X.shape)
print(y.shape)

# train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test)


# ------Model training----------
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)


# ------Only One home prediction-------
single_pred = model.predict(X_test.iloc[[0]])
print("\t---One home Result----")
print(f"Predicted price: ${round(np.expm1(single_pred[0]))}")
print(f"Original price : ${round(np.expm1(y_test.iloc[0]))}")


# --------First 5 home prediction------------
result_5 = pd.DataFrame({
"Original Price": np.expm1(y_test.values[:5]).round(),
"model_pred": np.expm1(pred[:5]).round()
}
)
print(result_5)


# ------- Find the Error in models--------
mae = mean_absolute_error(y_test, pred)
print(f"Mean Absolute Error: {mae}")

mse = mean_squared_error(y_test, pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root mean squard error: {rmse}")

r2 = r2_score(y_test, pred)
print(f"r2_score: {r2}")


train_r2 = r2_score(y_train, model.predict(X_train))
print(f"Train R2: {train_r2}")
print(f"Test R2 : {r2}")



# ------- Metric Comparison -------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=30)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.005, max_iter=10000)),
}

rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    train_pred = m.predict(X_train)
    test_pred = m.predict(X_test)

    rows.append({
        "Model": name,
        "Train R2": r2_score(y_train, train_pred),
        "Test R2": r2_score(y_test, test_pred),
        "MAE": mean_absolute_error(y_test, test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "MAPE %": mean_absolute_percentage_error(np.expm1(y_test), np.expm1(test_pred)) * 100,
    })

comparison = pd.DataFrame(rows)
print(comparison.round(4).to_string(index=False))




# ------- Model Interpretation: Top features (Lasso) -------------
lasso = models["Lasso"].named_steps["lasso"]

coef = pd.Series(lasso.coef_, index=X.columns)
coef = coef[coef != 0].sort_values()       

top = pd.concat([coef.head(5), coef.tail(10)])   

colors = ["red" if v < 0 else "green" for v in top]
top.plot(kind="barh", color=colors, figsize=(8, 6))
plt.axvline(0, color="black")
plt.xlabel("Coefficient (price par asar)")
plt.title("Lasso: Top features")
plt.tight_layout()
plt.show()



# ------- Small prediction model-------------
f = ["OverallQual", "TotalSF", "GarageCars", "HouseAge", "TotalBath"]

small_model = LinearRegression()
small_model.fit(X_train[f], y_train)

# ----- User input -----
quality = int(input("Quality (1-10): "))
area    = float(input("Total area sq ft (330-6900): "))
cars    = int(input("Garage Cars (0-4): "))
age     = int(input("House Age: "))
bath    = float(input("Total bathrooms: "))


if not (1 <= quality <= 10 and 330 <= area <= 7000 and 0 <= cars <= 4):
    print("Wrong Input! Quality (1-10), area (330-7000), cars (0-4) .")
else:
    new = pd.DataFrame([[quality, area, cars, age, bath]], columns=f)
    price = np.expm1(small_model.predict(new))[0]
    print(f"Andazan price: ${round(price)}")



# --------Actuall and prediction Graph---------------
plt.scatter(y_test,pred, alpha= 0.2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color= 'red')
plt.xlabel("Actual Price (log)")
plt.ylabel("predicted Price (log)")
plt.title("Predicted vs Actual Prices")
plt.show()


# ---------Residual Graph---------------
residuals = y_test - pred
plt.scatter(pred, residuals, alpha= 0.2)
plt.axhline(0, color="red")
plt.xlabel("Total predicted")
plt.ylabel("Total residuals")
plt.title("Residual plot")
plt.show()


# -----------Residual Histogram-----------
plt.hist(residuals, bins=30, edgecolor="black")
plt.xlabel("Residual")
plt.ylabel("Total Values")
plt.title("Residuals Distribution")
plt.show()



# ---------Before & after SalePrice Log-----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(data["SalePrice"], bins=30)
ax[0].set_title("Original SalePrice ")

ax[1].hist(np.log1p(data["SalePrice"]), bins=30)
ax[1].set_title("After Log ")

plt.show()


print("==================================")
print("==          End                 ==")
print("==================================")


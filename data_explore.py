# %%
from turtle import title
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as stm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score, mean_squared_error

# %%
COLS = [f"x_{i}" for i in range(1, 9)]
train_data = pd.read_csv(
    "data/traindata.txt", sep="\s+", header=None, names=COLS + ["y"]
)
test_data = pd.read_csv("data/testinputs.txt", sep="\s+", header=None, names=COLS)

train_data.shape  # , test_data.shape


# %%
train_data.info()
# %%
test_data.info()

# %%
train_data_X = train_data.drop("y", axis=1)
train_data_Y = train_data["y"]

# %%

train_x, val_x, train_y, val_y = train_test_split(
    train_data_X, train_data_Y, random_state=42
)

# %%

# lr = LinearRegression()
# lr = Ridge()
# lr = Lasso()
lr = DecisionTreeRegressor(random_state=42)
# lr = ExtraTreeRegressor(random_state=42)
lr = RandomForestRegressor(random_state=42)
lr = MLPRegressor(
    hidden_layer_sizes=(24),
    solver="lbfgs",
    random_state=42,
    learning_rate="adaptive",
    learning_rate_init=1e-5,
    max_iter=10000,
    # verbose=True,
)

cols_to_drop = ["x_6", "x_7"]
# cols_to_drop = []


lr.fit(train_x.drop(cols_to_drop, axis=1), train_y)

train_pred_y = lr.predict(train_x.drop(cols_to_drop, axis=1))
y_pred = lr.predict(val_x.drop(cols_to_drop, axis=1))

print("mse = ", mean_squared_error(train_y, train_pred_y))
print("r2 = ", r2_score(train_y, train_pred_y))

print("mse = ", mean_squared_error(val_y, y_pred))
print("r2 = ", r2_score(val_y, y_pred))


# %%
sns.pairplot(train_data)
# %%

fig, ax = plt.subplots(2, 4)
for i, feat in enumerate(COLS):
    row, col = divmod(i, 4)
    train_data.plot(x=feat, y="y", ax=ax[row][col], kind="scatter", s=3)


# %%


train_data.assign(x_9=(train_data["x_4"] + train_data["x_8"])).plot(
    x="x_9", y="y", kind="scatter", s=2
)

# %%
lr = stm.OLS(train_data["y"], stm.add_constant(train_data.drop(["y"], 1)))
# lr = stm.OLS(train_data["y"], stm.add_constant(train_data.drop(["x_6", "x_7", "y"], 1)))

fit_lr = lr.fit()
fit_lr.summary()


# %%

# norm_train_data = (train_data - train_data.mean()) / train_data.std()
norm_train_data = (train_data - train_data.min()) / (
    train_data.max() - train_data.min()
)

# %%
sns.pairplot(norm_train_data)
# %%


norm_train_data.assign(x_1=(norm_train_data["x_1"] ** 1)).plot(
    x="x_1", y="y", kind="scatter", s=2
)

# %%
lr_n = stm.OLS(norm_train_data["y"], stm.add_constant(norm_train_data.drop(["y"], 1)))
# lr = stm.OLS(norm_train_data["y"], stm.add_constant(norm_train_data.drop(["x_6", "x_7", "y"], 1)))

fit_lr_n = lr_n.fit()
fit_lr_n.summary()

# %%

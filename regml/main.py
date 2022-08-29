import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes()
#
diabetes_X = diabetes.data[:, np.newaxis, 2]
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.keys())
# print(diabetes.DESCR)
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

# Line obtained through training

model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

print("mean squared error is : ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
# prdicted line through diabetes_Y_predicted

print("Weights: ", model.coef_)
print("intercept: ", model.intercept_)

plt.xlabel("X")
plt.ylabel("Y")

plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predicted)
plt.show()

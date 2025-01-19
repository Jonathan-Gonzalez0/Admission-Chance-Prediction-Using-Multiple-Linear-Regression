# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:42:17 2025

@author: Jonathan Gonzalez


This Python script analyzes and predicts admission chances 
using academic and application metrics. It preprocesses the 
data, visualizes relationships through EDA, and builds a 
multiple linear regression models to evaluate predictive 
performance. 

Last Updated: 1/18/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

admissionDataSet = pd.read_csv("Admission.csv")

print(admissionDataSet.head(5))

print(admissionDataSet.tail(10))

print(admissionDataSet.describe())

print(admissionDataSet.info())

admissionDataSet = admissionDataSet.drop(['Serial No.'], axis = 1)

columnHeaders = admissionDataSet.columns.values

plt.close("all")
i = 1
fig, ax = plt.subplots(2,4,figsize = (10,10))
for headers in columnHeaders:
    plt.subplot(2,4,i)
    sns.boxplot(admissionDataSet[headers])
    i += 1

i = 1
fig, ax = plt.subplots(2,4,figsize = (10,10))
for headers in columnHeaders:
    plt.subplot(2,4,i)
    sns.distplot(admissionDataSet[headers])
    i += 1

plt.figure(figsize = (10,10))
sns.heatmap(admissionDataSet.corr(), annot = True)

sns.pairplot(admissionDataSet)

plt.figure()
sns.heatmap(admissionDataSet.isnull())

x = admissionDataSet.drop(["Admission Chance"], axis = 1)

y = admissionDataSet["Admission Chance"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(x_train,y_train)

print(f"Linear Model Coeff (m), {regressor.coef_}")
print(f"Linear Model Coeff (b), {regressor.intercept_}")

y_predict = regressor.predict(x_test)

plt.figure()
plt.scatter(y_test,y_predict, color = "red")
plt.title("Using 7 Variables", weight = "bold", size = 20)
plt.ylabel("Model Prediction", size = 15)
plt.xlabel("True (Ground truth)", size = 15)

plt.figure()
plt.title("Admission Chance VS GRE Score (Using 7 Variables)", weight = "bold", size = 15)
plt.plot(x_test["GRE Score"], y_test, "o", color = "b")
plt.plot(x_test["GRE Score"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("GRE Score", size = 15)

plt.figure()
plt.title("Admission Chance VS TOEFL Score (Using 7 Variables)", weight = "bold", size = 14)
plt.plot(x_test["TOEFL Score"], y_test, "o", color = "b")
plt.plot(x_test["TOEFL Score"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("TOEFL Score", size = 15)

plt.figure()
plt.title("Admission Chance VS University Rating", weight = "bold", size = 17)
plt.plot(x_test["University Rating"], y_test, "o", color = "b")
plt.plot(x_test["University Rating"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("University Rating", size = 15)

plt.figure()
plt.title("Admission Chance VS SOP", weight = "bold", size = 17)
plt.plot(x_test["SOP"], y_test, "o", color = "b")
plt.plot(x_test["SOP"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("SOP", size = 15)

plt.figure()
plt.title("Admission Chance VS LOR", weight = "bold", size = 17)
plt.plot(x_test["LOR "], y_test, "o", color = "b")
plt.plot(x_test["LOR "], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("LOR", size = 15)

plt.figure()
plt.title("Admission Chance VS CGPA", weight = "bold", size = 17)
plt.plot(x_test["CGPA"], y_test, "o", color = "b")
plt.plot(x_test["CGPA"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("CGPA", size = 15)

plt.figure()
plt.title("Admission Chance VS Research", weight = "bold", size = 17)
plt.plot(x_test["Research"], y_test, "o", color = "b")
plt.plot(x_test["Research"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("CGPA", size = 15)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

k = x_test.shape[1]
n = len(x_test)

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)) , '.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
MAPE = np.mean( np.abs((y_test - y_predict) /y_test ) ) * 100
print("\n7 Variables:")
print("RMSE = ", RMSE, "\nMSE =", MSE, "\nMAE =", MAE, "\nR2 =", r2, "\nAdjusted R2 =", adj_r2, "\nMAPE =", MAPE, "%")

x = x[["GRE Score", "TOEFL Score"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

plt.figure()
plt.scatter(y_test,y_predict, color = "red")
plt.title("Using 2 Variables")
plt.ylabel("Model Prediction")
plt.xlabel("True (Ground truth)")

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)) , '.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
MAPE = np.mean( np.abs((y_test - y_predict) /y_test ) ) * 100
print("\n2 Variables:")
print("RMSE = ", RMSE, "\nMSE =", MSE, "\nMAE =", MAE, "\nR2 =", r2, "\nAdjusted R2 =", adj_r2, "\nMAPE =", MAPE, "%")

from mpl_toolkits.mplot3d import Axes3D

x_surf, y_surf = np.meshgrid(np.linspace(admissionDataSet["GRE Score"].min(), admissionDataSet["GRE Score"].max(), 100) , np.linspace(admissionDataSet["TOEFL Score"].min(), admissionDataSet["TOEFL Score"].max(), 100) )

onlyX = pd.DataFrame({"GRE Score": x_surf.ravel(), "TOEFL Score": y_surf.ravel()})

fittedY = regressor.predict(onlyX)

fittedY = fittedY.reshape(x_surf.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

ax.scatter(admissionDataSet["GRE Score"], admissionDataSet["TOEFL Score"], admissionDataSet["Admission Chance"])
ax.plot_surface(x_surf,y_surf,fittedY, color = "red")
ax.set_xlabel("GRE Score")
ax.set_ylabel("TOEFL Score")
ax.set_zlabel("Admission Chance")

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

ax.scatter(x_test["GRE Score"],x_test["TOEFL Score"] , y_test)
ax.scatter(x_test["GRE Score"],x_test["TOEFL Score"] , y_predict)
ax.set_xlabel("GRE Score")
ax.set_ylabel("TOEFL Score")
ax.set_zlabel("Admission Chance")

plt.figure()
plt.title("Admission Chance VS GRE Score (Using 2 Variables)", weight = "bold", size = 15)
plt.plot(x_test["GRE Score"], y_test, "o", color = "b")
plt.plot(x_test["GRE Score"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("GRE Score", size = 15)

plt.figure()
plt.title("Admission Chance VS TOEFL Score (Using 2 Variables)", weight = "bold", size = 14)
plt.plot(x_test["TOEFL Score"], y_test, "o", color = "b")
plt.plot(x_test["TOEFL Score"], y_predict, "o", color = "r")
plt.ylabel("Admission Chance", size = 15)
plt.xlabel("TOEFL Score", size = 15)
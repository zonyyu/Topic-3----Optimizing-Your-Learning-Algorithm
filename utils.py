import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from bokeh.plotting import show, figure, output_notebook
from bokeh.layouts import column

x = np.array([0.87, 2.01, 1.41, 2.31, 3.65, 4.39, 5.87]).reshape(-1, 1)
y = np.array([1.67, 2.65, 2.91, 3.55, 3.91, 3.65, 2.85])
x_test = np.array([6.3, 3.14, 5.32, 4.23, 2.22, 1.07]).reshape(-1, 1)
y_test = np.array([2.18, 3.69, 3.40, 4.01, 2.98, 2.27])

def disp_underfit():
    model = LinearRegression()
    model.fit(x, y)
    p = figure(width=600, height=400, x_range=(0, 7), y_range=(0, 5), x_axis_label="Input", y_axis_label="Output", title="Degree 1 Polynomial" )
    p.circle(x.flatten(), y, color="blue", legend_label="Training")

    x_pred = np.arange(0, 7, 0.1).reshape(-1, 1)
    y_pred = model.predict(x_pred)
    p.line(x_pred.flatten(), y_pred, color='orange')

    p.circle(x_test.flatten(), y_test, color="red", legend_label="Testing")

    p.legend.location = "top_left"
    # add a title to your legend
    p.legend.title = "Data Points"

    output_notebook()
    show(p) 


def disp_overfit():
    model = LinearRegression()
    poly = PolynomialFeatures(degree = 10)
    xp = poly.fit_transform(x)
    model.fit(xp, y)
    p = figure(width=600, height=400, x_range=(0, 7), y_range=(0, 5), x_axis_label="Input", y_axis_label="Output", title="Degree 10 Polynomial" )
    p.circle(x.flatten(), y, color="blue", legend_label="Training")

    x_pred = np.arange(0, 7, 0.1).reshape(-1, 1)
    x_predp = poly.transform(x_pred)
    y_pred = model.predict(x_predp)
    p.line(x_pred.flatten(), y_pred, color='orange')

    p.circle(x_test.flatten(), y_test, color="red", legend_label="Testing")

    p.legend.location = "top_left"
    # add a title to your legend
    p.legend.title = "Data Points"


    output_notebook()
    show(p) 

def disp_good_fit():

    model = LinearRegression()
    poly = PolynomialFeatures(degree = 3)
    xp = poly.fit_transform(x)
    model.fit(xp, y)
    p = figure(width=600, height=400, x_range=(0, 7), y_range=(0, 5), x_axis_label="Input", y_axis_label="Output", title="Degree 3 Polynomial" )
    p.circle(x.flatten(), y, color="blue", legend_label="Training")

    x_pred = np.arange(0, 7, 0.1).reshape(-1, 1)
    x_predp = poly.transform(x_pred)
    y_pred = model.predict(x_predp)
    p.line(x_pred.flatten(), y_pred, color='orange')

    p.circle(x_test.flatten(), y_test, color="red", legend_label="Testing")

    p.legend.location = "top_left"
    p.legend.title = "Data Points"

    output_notebook()
    show(p) 




def disp_cost_plots():
    x = np.arange(0, 100, 0.2)
    p1 = figure(width=600, height=400, x_range=(0, 100), y_range=(0, 2), x_axis_label="Iterations", y_axis_label="Cost", title="High Bias (Underfit)")
    y1_train = 1/(x + 0.3)+0.5
    y1_test = 1/(x + 0.3)+0.7
    p1.line(x, y1_train, color="blue", legend_label="Training Cost")
    p1.line(x, y1_test, color="red", legend_label="Validation Cost")
    p1.legend.location = "top_right"
    p1.legend.title = "Costs"

    p2 = figure(width=600, height=400, x_range=(0, 100), y_range=(0, 2), x_axis_label="Iterations", y_axis_label="Cost", title="High Variance (Overfit)")
    y2_train = -np.log10((x + 3)/103)
    y2_test = 1/50 * ((x-50)/10)**2 / ((x-50)/100 +1) + 0.7
    p2.line(x, y2_train, color="blue", legend_label="Training Cost")
    p2.line(x, y2_test, color="red", legend_label="Validation Cost")
    p2.legend.location = "top_right"
    p2.legend.title = "Costs"

    p3 = figure(width=600, height=400, x_range=(0, 100), y_range=(0, 2), x_axis_label="Iterations", y_axis_label="Cost", title="Proper fit")
    y3_train = 5/np.sqrt(x+5) - 0.4
    y3_test = 5/np.sqrt(x+5) - 0.3
    p3.line(x, y3_train, color="blue", legend_label="Training Cost")
    p3.line(x, y3_test, color="red", legend_label="Validation Cost")
    p3.legend.location = "top_right"
    p3.legend.title = "Costs"
    
    output_notebook()   
    show(column(p1, p2, p3))


def disp_reg(L2=1e2):

    model = LinearRegression()
    modelr = Ridge(alpha=L2)
    poly = PolynomialFeatures(degree = 10)
    xp = poly.fit_transform(x)
    model.fit(xp, y)
    p = figure(width=700, height=500, x_range=(0, 7), y_range=(0, 5), x_axis_label="Input", y_axis_label="Output", title="Degree 10 Polynomial with L2 vs without" )
    p.circle(x.flatten(), y, color="blue")

    x_pred = np.arange(0, 7, 0.01).reshape(-1, 1)
    x_predp = poly.transform(x_pred)
    y_pred = model.predict(x_predp)
    p.line(x_pred.flatten(), y_pred, color='orange', legend_label="Degree 10 Polynomial")

    p.circle(x_test.flatten(), y_test, color="red")

    p.legend.location = "top_left"
    # add a title to your legend
    p.legend.title = "Regularization"

    x_pr = poly.fit_transform(x)
    modelr.fit(x_pr, y)

    x_pred_r = np.arange(0, 7, 0.01).reshape(-1, 1)
    y_pred_r = modelr.predict(poly.transform(x_pred_r))
    p.line(x_pred_r.flatten(), y_pred_r, color="green", legend_label="Degree 10 Polynomial with L2")

    output_notebook()
    show(p)

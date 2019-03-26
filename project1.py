import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os


def main():
    data, row_count, col_count = load_data()
    
    x_features = data[:, 0:col_count - 1]
    y_labels = data[:, col_count - 1]

    reg = LinearRegression()
    reg = reg.fit(x_features, y_labels)

    y_predicted = reg.predict(x_features)
    sse = reg._residues

    clear_terminal()
    print('All features coefficients:', reg.coef_)
    print('Intercept:', reg.intercept_)
    print('SSE all features:', sse)
    print('MSE all features:', np.mean(np.square(np.subtract(y_labels, y_predicted))))

    #generate_scatter_all(x_features, y_labels)
    

def load_data():
    data = np.loadtxt('reDataUCI.csv', delimiter=",", skiprows=1)
    row_count = np.size(data, 0)
    col_count = np.size(data, 1)

    return (data, row_count, col_count)

def generate_scatter(title, y_label, x_label, index, x_features, y_labels):
    plt.figure()
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.scatter(x_features[:, index], y_labels)

def generate_scatter_all(x_features, y_labels):
    generate_scatter('ID Number vs.Y', 'Y label', 'ID Number', 0, x_features, y_labels)
    generate_scatter('Transaction Date vs.Y', 'Y label', 'Transaction Date', 1, x_features, y_labels)
    generate_scatter('House Age vs.Y', 'Y label', 'House Age', 2, x_features, y_labels)
    generate_scatter('Dist to MRT vs.Y', 'Y label', 'Dist to MRT', 3, x_features, y_labels)
    generate_scatter('Number of convenience stores vs.Y', 'Y label', 'Number of convenience stores', 4, x_features, y_labels)
    generate_scatter('Lat Coord vs.Y', 'Y label', 'Lat Coord', 5, x_features, y_labels)
    generate_scatter('Long coord vs.Y', 'Y label', 'Long coord', 6, x_features, y_labels)
    plt.show()

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

main()

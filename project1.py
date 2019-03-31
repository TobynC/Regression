import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os


def main():
    clear_terminal()
    #read in file
    data, row_count, col_count = load_data()
    
    #set up features and labels
    x_features = data[:, 0:col_count - 1]
    y_labels = data[:, col_count - 1]

    #create linear regression object and fit it against all features
    reg = LinearRegression()
    reg = reg.fit(x_features, y_labels)

    y_predicted = reg.predict(x_features)
    sse = reg._residues

    
    print('All features coefficients:')
    print_array(reg.coef_)
    print('Intercept:', reg.intercept_)
    print('SSE all features:', sse)
    print('MSE all features:', np.mean(np.square(np.subtract(y_labels, y_predicted))))

    # single_feature_plot(x_features, y_labels)
    # plt.figure()
    # plt.scatter(np.log10(np.sqrt(x_features[:, 2])), y_labels%100, color='g')
    

#     for i in range(7):
#         print('\n')
#         single_feature_plot(x_features, y_labels, i)

    plt.show()

#     #correlation coefficients
#     correlation = np.corrcoef(x_features, rowvar=False)
#     plt.imshow(correlation, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.show()

#     #remove id numbers
#     x_features_without_id = np.hstack((x_features[:, [1]], x_features[:, [2]], x_features[:, [3]], x_features[:, [4]], x_features[:, [5]], x_features[:, [6]]))

#     #new linear regression without id numbers
#     reg = LinearRegression().fit(x_features_without_id, y_labels)
#     print('SSE for trimmed features:', reg._residues)

#     x1 = x_features[:, [1]]
#     x1 = x1 % 1
    

def load_data():
    data = np.loadtxt('reDataUCI.csv', delimiter=",", skiprows=1)
    data = remove_outliers(data, 20, 60)
    row_count = np.size(data, 0)
    col_count = np.size(data, 1)

    return (data, row_count, col_count)

def generate_scatter(title, y_label, x_label, index, x_features, y_labels):
    plt.figure()
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.scatter(x_features[:, index], y_labels)

def single_feature_plot(x_features, y_labels, column_index):
    #generate_scatter_all(x_features, y_labels)
    reg = LinearRegression().fit(x_features[:, [column_index]], y_labels)
    predictedY = reg.predict(x_features[:, [column_index]])

    print("Coefficent for single variable: " + str(reg.coef_))
    print("SSE for single variable", column_index, ':', reg._residues)

    plt.figure()
    plt.scatter(x_features[:, column_index], y_labels, color='g')
    plt.plot(x_features[:, column_index], predictedY, color='r')

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

def print_array(arr):
    for i, item in enumerate(arr):
        print(str(i)+':', item)

def remove_outliers(data, low, high):
    print('old:', len(data))
    i = 0
    loopcount = len(data)
    while i < loopcount:
        if(data[i][7] > high or data[i][7] < low):
            data = np.delete(data, i, axis=0)
            loopcount-=1
        i+=1

    print('new:', len(data))

    return data

main()

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

    #remove id's
    x_features = np.hstack(
        (x_features[:, [1]], x_features[:, [2]], x_features[:, [3]], x_features[:, [4]], x_features[:, [5]], x_features[:, [6]])
    )

    #data transforms
    x_features[:,1] = np.sqrt(x_features[:,1])
    x_features[:,2] = np.log10(x_features[:,2])

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

    #correlation coefficients
    correlation = np.corrcoef(x_features, rowvar=False)
    plt.imshow(correlation, cmap='hot', interpolation='nearest')
    plt.colorbar()

    for i in range(6):
        plt.figure()
        x = x_features[:, [i]]
        reg = LinearRegression().fit(x, y_labels)
        predictedY = reg.predict(x)
        plt.scatter(x, y_labels, color='g')
        plt.plot(x, predictedY, color='r')

    plt.show()
    
    betas = ordinary_least_squares(x_features, y_labels)
    y = np.matmul(x_features, betas)
    reg = LinearRegression().fit(x_features, y)
    y_predicted = reg.predict(x_features)
    sse = np.sum(np.power(y_predicted - y_labels, 2))
    mse = sse/np.size(y_predicted)

    print("\nResults from OLS:\n")
    print("Coefficients:")
    print_array(betas)
    print('SSE:', sse)
    print('MSE:', mse)

    thetas = bgd(x_features, y_labels, 1e-7, 100, 1000)
    m_bgd = x_features.shape[0]
    x_features_ones = np.hstack((np.ones((m_bgd, 1)), x_features))

    y = np.matmul(x_features_ones, thetas)
    reg = LinearRegression().fit(x_features_ones, y)
    y_predicted = reg.predict(x_features_ones)
    sse = np.sum(np.power(y_predicted - y_labels, 2))
    mse = sse/np.size(y_predicted)

    print("\nResults from BGD:\n")
    print("Coefficients:")
    print_array(thetas)
    print('SSE:', sse)
    print('MSE:', mse, '\n')

    print('\n########################second data set##########################\n')

    #read in file
    data, row_count, col_count = load_secondary_data()

    #set up features and labels
    x_features = data[:, 0:col_count - 1]
    y_labels = data[:, col_count - 1]

    #data transforms
    #none

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

    #correlation coefficients
    correlation = np.corrcoef(x_features, rowvar=False)
    plt.imshow(correlation, cmap='hot', interpolation='nearest')
    plt.colorbar()
    for i in range(11):
        plt.figure()
        x = x_features[:, [i]]
        reg = LinearRegression().fit(x, y_labels)
        predictedY = reg.predict(x)
        plt.scatter(x, y_labels, color='g')
        plt.plot(x, predictedY, color='r')

    betas = ordinary_least_squares(x_features, y_labels)
    y = np.matmul(x_features, betas)
    reg = LinearRegression().fit(x_features, y)
    y_predicted = reg.predict(x_features)
    sse = np.sum(np.power(y_predicted - y_labels, 2))
    mse = sse/np.size(y_predicted)

    print("\nResults from OLS:\n")
    print("Coefficients:")
    print_array(betas)
    print('SSE:', sse)
    print('MSE:', mse)

    plt.show()
    thetas = bgd(x_features, y_labels, 0.0001, 60, 30)
    m_bgd = x_features.shape[0]
    x_features_ones = np.hstack((np.ones((m_bgd, 1)), x_features))

    y = np.matmul(x_features_ones, thetas)
    reg = LinearRegression().fit(x_features_ones, y)
    y_predicted = reg.predict(x_features_ones)
    sse = np.sum(np.power(y_predicted - y_labels, 2))
    mse = sse/np.size(y_predicted)

    print("\nResults from BGD:\n")
    print("Coefficients:")
    print_array(thetas)
    print('SSE:', sse)
    print('MSE:', mse)

def load_data():
    data = np.loadtxt('reDataUCI.csv', delimiter=",", skiprows=1)
    #remove outliers to keep only within range
    data = remove_outliers_from_y(data, 10, 60)
    row_count = np.size(data, 0)
    col_count = np.size(data, 1)

    return (data, row_count, col_count)

def load_secondary_data():
    data = np.loadtxt('winequality-red.csv', delimiter=';', skiprows=1)
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
    plt.show()

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_array(arr):
    for i, item in enumerate(arr):
        print(str(i)+':', item)

def remove_outliers_from_y(data, low, high):   
    i = 0
    loopcount = len(data)
    while i < loopcount:
        if(data[i][7] > high or data[i][7] < low):
            data = np.delete(data, i, axis=0)
            loopcount-=1
        i+=1
    return data

def ordinary_least_squares(training_data, labels):
    x_trans = np.transpose(training_data)
    betas = np.matmul(np.linalg.inv(np.matmul(x_trans, training_data)), np.matmul(x_trans, labels))  

    return betas

def bgd(training_data, labels, alpha, epsilon, epochs):
    m = training_data.shape[0]
    training_data = np.hstack((np.ones((m, 1)), training_data))

    thetas = np.random.random(training_data.shape[1])
    temp = []
    total_cost = 0

    for epoch in range(epochs):
        temp = []
        for theta in thetas:
            total_cost = 0
            for i, row in enumerate(training_data):
                predicted_y = np.dot(thetas, row)
                for j, data in enumerate(row):
                    label = labels[i]
                    total_cost += (predicted_y - label) * data
            tempi = theta - alpha * (1.0/m) * total_cost
            temp.append(tempi)

        #check for convergence        
        thetas = temp        
        if(abs(total_cost) <= epsilon): break

    return np.array(thetas)
    
main()

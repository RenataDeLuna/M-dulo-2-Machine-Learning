import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def linearRegr(X_train, y_train, learning_rate, epochs): 

    """
    Function that creates a linear regression model. 
    Inputs: 
    - X_train: Part of the DF that is going to be used to train the model.
    - y_train: Part of the DF that is going to train de model. Output/dependent values. 
    - Learning rate: Percentage of the weigths update for each iteration. 
    - Epochs: number of iterations

    Output: 
    - m: weight. 
    - c: bias.
    """

    m = 0 # Init value for weight
    c = 0 # Init value for bias

    n = float(len(X_train)) # Size of the df

    for j in range(epochs):
        for i in range(len(X_train)):
            y_pred = m * X_train + c
            d_m = (- 2 / n) * sum(X_train * (y_train - y_pred))
            d_c = (- 2 / n) * sum(y_train - y_pred)
            m = m - learning_rate * d_m
            c = c - learning_rate * d_c
    return m, c

def predictions(X_train, y_train, learning_rate, epochs, X_test):
    """
    Function that makes predictions with the model previuosly constructed. 
    Inputs: 
    - X_train: Part of the DF that is going to be used to train the model.
    - y_train: Part of the DF that is going to train de model. Output/dependent values. 
    - Learning rate: Percentage of the weigths update for each iteration. 
    - Epochs: number of iterations
    - X_test: part of the DF that is going to be used for the testing of the model. Input values. 

    Output: 
    - y_hat: the value of Y predicted.
    """
    m, c = linearRegr(X_train, y_train, learning_rate, epochs) 
    y_pred = m * X_test + c
    return y_pred

def errors(y_test, y_hat): 
    """
    Function that calculates the errors of the model. 
    Inputs: 
    - y_test: the original output value.
    - y_hat: the predicted value of y. 

    Output: 
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Percent Absolute Error
    - SSE: Sum of Squared Error
    """
    mae = sum(np.absolute(y_test - y_hat))
    rmse = np.sqrt(sum((y_test - y_hat) ** 2))
    mape = sum(np.absolute((y_test - y_hat) / y_test))
    sse = sum((y_test - y_hat) ** 2)
    return mae, rmse, mape, sse

def main():

    df = pd.read_csv('Sickness.csv') # Reading the input csv
    df_x = df.iloc[:,0] # Selecting the values for x
    df_y = df.iloc[:,1] # Selecting the values of y
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.25) # Spliting the dataframe for test and train

   #    Example 1
    print("Example 1")
    y_hat = predictions(X_train, y_train, 0.00001, 100, X_test) # Small lerning rate and epochs
    print(y_hat)
    print(y_test)

    mae, rmse, mape, sse = errors(y_test, y_hat)
    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("MAPE: ", mape)
    print("SSE: ", sse)

    r2 = r2_score(y_test, y_hat) # Squared error
    print("Model's R^2: ", r2)

    # Plotting
    plt.scatter(X_train, y_train) 
    plt.plot([min(X_train), max(X_train)], [min(y_hat), max(y_hat)], color='red')
    plt.show()

    # Example 2
    print("Example 2")
    
    y_hat2 = predictions(X_train, y_train, 0.00006, 500, X_test) # Bigger epochs and learning rate
    print(y_hat2)
    print(y_test)

    mae2, rmse2, mape2, sse2 = errors(y_test, y_hat2)
    print("MAE: ", mae2)
    print("RMSE: ", rmse2)
    print("MAPE: ", mape2)
    print("SSE: ", sse2)

    r2_2 = r2_score(y_test, y_hat2)
    print("Model's R^2: ", r2_2)


    plt.scatter(X_train, y_train) 
    plt.plot([min(X_train), max(X_train)], [min(y_hat2), max(y_hat2)], color='red')
    plt.show()

    # Example 3
    print("Example 3")
    y_hat3 = predictions(X_train, y_train, 0.000001, 1000, X_test) # Smaller learning rate and bigger epochs
    print(y_hat3)
    print(y_test)

    mae3, rmse3, mape3, sse3 = errors(y_test, y_hat3)
    print("MAE: ", mae3)
    print("RMSE: ", rmse3)
    print("MAPE: ", mape3)
    print("SSE: ", sse3)

    r2_3 = r2_score(y_test, y_hat3)
    print("Model's R^2: ", r2_3)


    plt.scatter(X_train, y_train) 
    plt.plot([min(X_train), max(X_train)], [min(y_hat3), max(y_hat)], color='red')
    plt.show()

    # Example 4
    print("Example 4")
    
    y_hat4 = predictions(X_train, y_train, 0.000008, 1500, X_test) # Bigger epochs and bigger learning rate
    print(y_hat4)
    print(y_test)

    mae4, rmse4, mape4, sse4 = errors(y_test, y_hat4)
    print("MAE: ", mae4)
    print("RMSE: ", rmse4)
    print("MAPE: ", mape4)
    print("SSE: ", sse4)

    r2_4 = r2_score(y_test, y_hat4)
    print("Model's R^2: ", r2_4)


    plt.scatter(X_train, y_train) 
    plt.plot([min(X_train), max(X_train)], [min(y_hat3), max(y_hat4)], color='red')
    plt.show()

    # Example 5
    # With a bigger learning rate, the model can not predict y_hat because the cvs has few values. 
    """print("Example 5")
    y_hat5 = predictions(X_train, y_train, 0.001, 2000, X_test) # Bigger epochs and smaller learning rate
    print(y_hat5)
    print(y_test)

    mae5, rmse5, mape5, sse5 = errors(y_test, y_hat5)
    print("MAE: ", mae5)
    print("RMSE: ", rmse5)
    print("MAPE: ", mape5)
    print("SSE: ", sse5)

    r2_5 = r2_score(y_test, y_hat5)
    print("Model's R^2: ", r2_5)


    plt.scatter(X_train, y_train) 
    plt.plot([min(X_train), max(X_train)], [min(y_hat5), max(y_hat5)], color='red')
    plt.show()"""


if __name__ == "__main__":
    main()





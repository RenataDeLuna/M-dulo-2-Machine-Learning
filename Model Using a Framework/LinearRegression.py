import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Library used to create the linear regression model
from sklearn.model_selection import train_test_split # Library used to split the dataset into train and test
from sklearn.metrics import mean_absolute_error,mean_squared_error # Library used to calculate mean squared error and mean error

regr = LinearRegression() # Name the function

def linearRegression(x_train, y_train, x_test, y_test): 
    regr.fit(x_train, y_train) # Model with train data
    score = regr.score(x_test, y_test) # Model's score
    return score

def predictions(x_test): 
    y_pred = regr.predict(x_test) # Make predictions with test data or input data
    return y_pred

def errors(y_test, y_pred):
    # Model's error
    mae = mean_absolute_error(y_test, y_pred)  
    mse = mean_squared_error(y_test, y_pred)
    return mae, mse

def inputPredictions(): 
    # Asking for information to make predictions
    age = int(input("Introduce the sick person's age: "))
    severity = int(input("Introduce the sickness severity: "))
    return age, severity

def main(): 

    # Reading the original csv
    df = pd.read_csv('Sickness.csv')

    # Selecting the dependent and independent values
    df_x = df.drop(['satisfaction'], axis=1)
    df_y = df['satisfaction']

    # Example 1

    # Spliting the data into train and test

    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.20)

    # Making the linear regression model and the calculation of its score
    score = linearRegression(X_train, y_train, X_test, y_test)
    print("The model's score is: ", str(score))

    print()

    # Obtaining the value of y_pred to compare it to the original value
    y_pred = predictions(X_test)
    y_pred = pd.DataFrame(y_pred)

    print('The predictions for the x test are: ', str(y_pred))
    print(y_test)

    print()

    # Obteining the errors of the model
    mae, mse = errors(y_test, y_pred)

    print("The model's errors are: ")
    print('MAE: ', str(mae))
    print('MSE: ', str(mse))

    print()

    # Dividing the DF into train/test samples, but bigger

    print("Example 2")

    print("Test size of 50%")

    X_train2, X_test2, y_train2, y_test2 = train_test_split(df_x, df_y, test_size = 0.50)
    
    score2 = linearRegression(X_train2, y_train2, X_test2, y_test2)
    print("The model's score is: ", str(score2))

    print()

    # Obtaining the value of y_pred to compare it to the original value
    y_pred2 = predictions(X_test2)
    y_pred2 = pd.DataFrame(y_pred2)

    print('The predictions for the x test are: ', str(y_pred2))
    print(y_test2)

    print()

    # Obteining the errors of the model
    mae2, mse2 = errors(y_test2, y_pred2)

    print("The model's errors are: ")
    print('MAE: ', str(mae2))
    print('MSE: ', str(mse2))

    # Dividing the DF into train/test samples, but smaller
    print("Example 3")

    print("Test size of 10%")

    X_train3, X_test3, y_train3, y_test3 = train_test_split(df_x, df_y, test_size = 0.10)
    
    score3 = linearRegression(X_train3, y_train3, X_test3, y_test3)
    print("The model's score is: ", str(score3))

    print()

    # Obtaining the value of y_pred to compare it to the original value
    y_pred3 = predictions(X_test3)
    y_pred3 = pd.DataFrame(y_pred3)

    print('The predictions for the x test are: ', str(y_pred3))
    print(y_test3)

    print()

    # Obteining the errors of the model
    mae3, mse3 = errors(y_test3, y_pred3)

    print("The model's errors are: ")
    print('MAE: ', str(mae3))
    print('MSE: ', str(mse3))


    # Making predictions with other values

    print("Making predictions with input data:")
    age, severity = inputPredictions()
    df_pred = pd.DataFrame(data=[[age, severity]], columns=["age", "severity"])
    y_new_pred = predictions(df_pred)
    print()
    print("Satisfaction: ", y_new_pred[0])

if __name__ == "__main__":
    main()
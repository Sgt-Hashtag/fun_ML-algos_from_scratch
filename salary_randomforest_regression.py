import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeRegressor
import sklearn
import warnings
import numpy as np

from sklearn.preprocessing import LabelEncoder # encode categorical data into numerical values.
from sklearn.impute import KNNImputer # to impute missing values in a dataset using a k-nearest neighbors approach
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor # regression model that is based upon the Random Forest model. 
from sklearn.model_selection import cross_val_score # k-fold cross-validation

warnings.filterwarnings('ignore')
#random forest has capability to handle high-dimensional data, capture complex relationships and reduce overfitting
#could have used my kmeans from scratch nbut that would be nuts
def main(Rndomforest):
    df= pd.read_csv('Salaries.csv')
    print(df.info())
    X = df.iloc[:,1:-1].values
    y = df.iloc[:,-1].values
    
    label_encoder = LabelEncoder()
    x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    x_numerical = df.select_dtypes(exclude=['object']).values
    x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values #Combines the numerical and encoded categorical features horizontally
        
        
    if Rndomforest:
        
        regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True) # estimators are number of decision trees
        regressor.fit(x,y)
        oob_score = regressor.oob_score_
        print(f'Out-of-Bag Score: {oob_score}')
    
        #predictions
        predictions = regressor.predict(x)
        
        mse = mean_squared_error(y, predictions)
        print(f'Mean Squared Error: {mse}')
        
        r2 = r2_score(y, predictions)
        print(f'R-squared: {r2}')
        
        tree_to_plot = regressor.estimators_[0]
        
        plt.figure(figsize=(20, 10))
        plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree from Random Forest")
        plt.show()
    else:
        regressor = DecisionTreeRegressor( random_state = 0)
        regressor.fit(X,y)
    
        #predictions
        predictions = regressor.predict(X)

        # Predicting a new result
        regressor.predict([[6.5]])

        # Visualising the Decision Tree Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Decision Tree Regression')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()
    
if __name__ == "__main__":
    Rndomforest =True
    main(Rndomforest)
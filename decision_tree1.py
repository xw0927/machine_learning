import pandas as pd
melbourne_data=pd.read_csv("../input/train.csv")
print(melbourne_data.describe())
two_columns=melbourne_data[['LotArea','SalePrice']]
two_columns.describe()

y=melbourne_data.SalePrice
x=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

X=melbourne_data[x]
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)
#Define model
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X,train_y)

predict_s=melbourne_model.predict(val_X)
mean_absolute_error(val_y,predict_s)

def get_mae(max_leaf_nodes,predictors_train,predictors_val,targ_train,targ_val):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(predictors_train,targ_train)
    preds_val=model.predict(predictors_val)
    mae=mean_absolute_error(targ_val,preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5,50,500,5000]:
    my_mae=get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

import numpy as np
import pandas as pd
from time import time


# Import supplementary visualization code visuals.py
import visuals3 as vs

# Pretty display for notebooks

# Load the Census dataset
data = pd.read_csv("data.csv")

n_records=data.shape[0]

n_greater_50k = len(data[data.income=='>50K'])
n_at_most_50k = len(data[data.income=='<=50K'])
greater_percent = (n_greater_50k*100)/n_records

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

vs.distribution(data)
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)

from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))

features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 0 if x== '<=50K' else 1)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
#encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case


# TODO: Calculate accuracy, precision and recall
accuracy = (TP)/(TP+FP+TN+FN)
recall = (TP)/(TP+FN)
precision = (TP)/(TP+FP)
# Calculating the  F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta=0.5
fscore = (1 + beta** 2)*(precision * recall)/(beta** 2 *precision + recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import fbeta_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
            
    # TODO: Compute accuracy on the first 300 training samples
    from sklearn.metrics import accuracy_score
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,beta=0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)
    print(results)

sample_sizes=(362,3618,36177,362,3618,36177,362,3618,36177)
learners=(AdaBoostClassifier(),AdaBoostClassifier(),AdaBoostClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),SVC(),SVC(),SVC())
for i,j in zip(sample_sizes,learners):
    train_predict(j, i, X_train, y_train, X_test, y_test)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


classifier=AdaBoostClassifier()
params={'n_estimators':[50,100,200,300],'learning_rate':[0.5,1.0,1.5,2.0]}
scorer=make_scorer(fbeta_score,beta=0.5)

grid_object=GridSearchCV(classifier, params,scoring=scorer)
grid_fit=grid_object.fit(X_train,y_train)
best_classifier=grid_fit.best_estimator_
best_prediction=best_classifier.predict(X_test)
print(best_classifier,best_prediction)
from sklearn.metrics import accuracy_score
print(accuracy_score(best_prediction,y_test))
print(best_classifier.feature_importances_)



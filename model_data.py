import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

train_data = pd.read_csv("../data_csv/train.csv")
test_data = pd.read_csv("../data_csv/test.csv")

# Split train to x and y
y = train_data["label"]
del train_data["label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=1)

# Scale data
scaler = MinMaxScaler()
# Model
clf = SVC(C=8)
# Pipeline
pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])

scores = cross_val_score(pipeline, X_train, y_train, cv=10)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf.fit(X_train, y_train)

X_test = scaler.transform(X_test)
cost_train = mean_squared_error(y_train, clf.predict(X_train))
cost_test = mean_squared_error(y_test, clf.predict(X_test))
print("Cost train: %f Cost test: %f" % (cost_train, cost_test))

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("Train score: %f Test score: %f" % (train_score, test_score))


# Fit unlabeled data
test_data = scaler.transform(test_data)
Y = pd.Series(clf.predict(test_data))
Y.name = "Label"
image_id = pd.Series(range(1, Y.shape[0] + 1))
image_id.name = "ImageId"
prediction_data = pd.concat([image_id, Y], axis=1)
# Create prediction csv
prediction_data.to_csv("../data_csv/prediction.csv", index=False)

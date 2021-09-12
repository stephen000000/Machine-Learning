import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from c45 import C45

data_file = "hazelnuts.txt"
feature_cols = ["length", "width", "thickness", "surface_area", "mass", "compactness", "hardness", "shell_top_radius", "water_content", "carbohydrate_content", "variety"]

data = pd.read_csv(data_file, sep="\t").T
data.columns = feature_cols

for i in feature_cols[0:-1]:
    data[i] = data[i].astype(np.float64)

X = data.drop("variety", axis=1)
y = data.variety

total_acc = 0
num_tests = 1

for i in range(num_tests):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)

    # dt = C45(X_train, y_train)
    # dt.generate_tree(False)
    # dt.print_tree()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    print(clf.predict(X_test))

#     acc = dt.predict(X_test, y_test)
#     print("Accuracy", i+1, ":", acc)
#     total_acc += acc

# print("\nAverage Accuracy :", total_acc/num_tests)
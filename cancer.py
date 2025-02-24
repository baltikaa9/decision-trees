import pydotplus
from six import StringIO
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = load_breast_cancer(return_X_y=True, as_frame=True)

X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на наборе данных Breast Cancer: {accuracy * 100:.2f}%")

dot_data = StringIO()
export_graphviz(
    clf,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=X.columns,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('clf.png')

for i in range(len(y_pred)):
    print(f'prediction: {y_pred[i]}, actual: {y_test.iloc[i]}')


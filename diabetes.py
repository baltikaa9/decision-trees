import pandas as pd
import pydotplus
from six import StringIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, DecisionTreeRegressor

df = pd.read_csv("diabetes.csv")

print(df.head())

X = df.drop(columns=["Glucose"])
y = df["Glucose"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
print(f"R2-оценка модели: {r2:.2f}")

dot_data = StringIO()
export_graphviz(
    reg,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=X.columns,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('reg.png')

for i in range(len(y_pred)):
    print(f'prediction: {y_pred[i]}, actual: {y_test.iloc[i]}')


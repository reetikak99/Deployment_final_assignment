import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
	# Load Iris dataset
	iris = load_iris()
	df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
	df['target'] = iris.target
	return df

def train_and_evaluate(data):
	X = data.drop(columns='target')
	y = data['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	model = RandomForestClassifier()
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	return accuracy_score(y_test, predictions)
def main():
	data = load_data()
	accuracy = train_and_evaluate(data)
	print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
	main()
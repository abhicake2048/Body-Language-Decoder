import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle



class BodyLanguageModel:
    def __init__(self, csv_file='coords.csv', test_size=0.1, random_state=1234):
        self.csv_file = csv_file
        self.test_size = test_size
        self.random_state = random_state

    def train_model(self):
        # Load the dataset
        df = pd.read_csv(self.csv_file)

        # Prepare features and target
        X = df.drop('class', axis=1)
        y = df['class']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        return model

    def save_model(self, model, file_name='body_language.pkl'):
        # Save the model to a file using pickle
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

def main():
    # Initialize the model trainer
    model_trainer = BodyLanguageModel()

    # Train the model
    trained_model = model_trainer.train_model()

    # Save the trained model to a file
    model_trainer.save_model(trained_model)

if __name__ == "__main__":
    main()

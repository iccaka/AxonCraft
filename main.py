import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Model
from layer import Layer
from initializer import Initializer

if __name__ == '__main__':
    cali = fetch_california_housing()
    X = cali.data
    y = cali.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Model([
        Layer(256, activation='relu', initializer=Initializer.xavier_norm),
        Layer(128, activation='relu', initializer=Initializer.xavier_norm),
        Layer(64, activation='relu', initializer=Initializer.xavier_norm),
        Layer(32, activation='relu', initializer=Initializer.xavier_norm),
        Layer(1, activation='linear', initializer=Initializer.xavier_norm)
    ])

    model.build(_input_shape=X_train.shape)
    model.configure(loss='mean_squared_error', learning_rate=0.1)
    cost = model.fit(X_train, y_train, epochs=30)

    plt.plot(range(len(cost)), cost)
    plt.show()

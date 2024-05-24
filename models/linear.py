import numpy as np
import matplotlib.pyplot as plt


class Linear:
    def __init__(self, n_class, threshold):
        self.w = None
        self.n_class = n_class
        self.threshold = threshold

    def add_bias(self, X):  # adding bias
        return np.column_stack([X, np.ones(X.shape[0])])

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.w = {}

        X_train_bias = self.add_bias(X_train)
        pinv_X = np.linalg.pinv(X_train_bias)

        for class_label in self.classes:
            y_binary = np.where(y_train == class_label, 1, -1)
            # closed form weights calculation
            self.w[class_label] = np.dot(pinv_X, y_binary)

    def predict(self, X_test, strategy='max', threshold=2):
        X_test_bias = self.add_bias(X_test)

        scores = {class_label: np.dot(
            X_test_bias, self.w[class_label]) for class_label in self.classes}

        if strategy == 'max':
            predictions = np.array(
                [max(scores, key=lambda k: scores[k][i]) for i in range(X_test.shape[0])])
        elif strategy == 'closest':
            predictions = np.array([min(scores, key=lambda k: abs(
                scores[k][i] - threshold)) for i in range(X_test.shape[0])])

        return predictions

    # plotting threshold values
    def plot_th(self, X_test, y_test, thresholds):
        accuracies = []
        for threshold in thresholds:
            self.threshold = threshold
            y_pred = self.predict(
                X_test, strategy="closest", threshold=threshold)
            accuracy = np.mean(y_pred == y_test)*100
            accuracies.append(accuracy)

        plt.plot(thresholds, accuracies, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Performance vs Threshold')
        plt.grid(True)
        plt.show()

    # printing weights for rice dataset
    def print_weights(self):
        for class_label, weights in self.w.items():
            print(f"Class {class_label} weights:")
            print(weights)

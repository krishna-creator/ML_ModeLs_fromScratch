import numpy as np
import matplotlib.pyplot as plt


class Logistic:
    def __init__(self, n_class, lr, epochs, threshold):
        self.w = None  # Weights
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.n_class = n_class

    def sigmoid(self, z):  # sigmoid function
        exp_vals = np.exp(np.clip(-z, -500, 500))
        return 1 / (1 + exp_vals)

    def model(self, X_train, y_train, lr, tr):
        accuracies = []
        self.classes = np.unique(y_train)
        self.w = {}

        for class_label in self.classes:
            self.w[class_label] = np.zeros(X_train.shape[1])
        for epoch in range(self.epochs):
            for class_label in self.classes:
                y_binary = np.where(y_train == class_label, 1, 0)

                for x, y in zip(X_train, y_binary):
                    y_pred = self.sigmoid(np.dot(x, self.w[class_label]))
                    # calulating gradients
                    gradient = np.dot(x.T, (y_pred - y))
                    self.w[class_label] -= lr * gradient

            y_pred_train = self.predict(X_train, tr)

            accuracy = np.mean(y_pred_train == y_train) * 100
            accuracies.append(accuracy)
        return accuracies

    def train(self, X_train, y_train):
        self.model(X_train, y_train, self.lr, self.threshold)
        return

    def predict(self, X_test, threshold=None):
        if threshold is None:
            threshold = self.threshold
        y_pred = []
        for x in X_test:
            scores = [np.dot(x, self.w[c]) for c in range(self.n_class)]
            if np.max(scores) >= threshold:
                predicted_class = np.argmax(scores)
            else:
                predicted_class = -1
            y_pred.append(predicted_class)
        return np.array(y_pred)
    # plotting epochs

    def plot_epoches(self, X_train, y_train):
        accuracy_epochs = self.model(
            X_train, y_train, self.lr, self.threshold)
        plt.plot(range(1, self.epochs+1), accuracy_epochs, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.grid(True)
        plt.show()
    # plotting learning rates

    def plot_lr(self, X_train, y_train):
        accuracy_lr = []
        for lr in self.lr:
            accuracies = self.model(X_train, y_train, lr, self.threshold)
            accuracy_lr.append(accuracies[-1])
        plt.plot(self.lr, accuracy_lr, marker='o')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Learning Rate')
        plt.grid(True)
        plt.show()
    # plotting thresholds

    def plot_tr(self, X_train, y_train):
        accuracy_tr = []
        for tr in self.threshold:
            accuracies = self.model(X_train, y_train, self.lr, tr)
            accuracy_tr.append(accuracies[-1])
        plt.plot(self.threshold, accuracy_tr, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True)
        plt.show()

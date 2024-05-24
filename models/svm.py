import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, n_class, lr, epochs, reg_const):
        self.w = None  # TODO
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train, y_train, lr, r):
        n = X_train.shape[0]
        for x, y in zip(X_train, y_train):
            y_pred = self.predict(x.reshape(1, -1))
            margins = np.dot(x, self.w.T) - np.dot(x, self.w[y_pred].T) + 1
            for c in range(self.n_class):
                if c == y:
                    continue
                if margins[c] > 0:  # if wrong class update weights
                    self.w[y] += lr * (r / n) * x
                    self.w[c] -= lr * (r / n) * x

    def model(self, X_train, y_train, lr, r):
        accuracies = []
        self.w = np.zeros((self.n_class, X_train.shape[1]))
        for i in range(self.epochs):
            self.calc_gradient(X_train, y_train, lr, r)
            y_pred_train = self.predict(X_train)
            accuracy = np.mean(y_pred_train == y_train) * 100
            accuracies.append(accuracy)
        return accuracies

    def train(self, X_train, y_train):
        self.model(X_train, y_train,
                   self.lr, self.reg_const)

    def predict(self, X_test) -> np.ndarray:
        scores = np.dot(X_test, self.w.T)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    # plotting epochs

    def plot_epoches(self, X_train, y_train):
        accuracy_epochs = self.model(
            X_train, y_train, self.lr, self.reg_const)
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
            accuracies = self.model(
                X_train, y_train, lr, self.reg_const)
            accuracy_lr.append(accuracies[-1])
        plt.plot(self.lr, accuracy_lr, marker='o')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Learning Rate')
        plt.grid(True)
        plt.show()
    # plotting regularization

    def plot_r(self, X_train, y_train):
        accuracy_r = []
        for r in self.reg_const:
            accuracies = self.model(
                X_train, y_train, self.lr, r)
            accuracy_r.append(accuracies[-1])
        plt.plot(self.reg_const, accuracy_r, marker='o')
        plt.xlabel('Reg_const')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Reg_const')
        plt.grid(True)
        plt.show()

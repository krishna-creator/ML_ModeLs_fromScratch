import numpy as np
import matplotlib.pyplot as plt


class Softmax:
    def __init__(self, n_class, lr, epochs, reg_const):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def softmax(self, scores):
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        return exp_scores / np.sum(exp_scores)

    def model(self, X_train, y_train, lr, r):
        accuracies = []
        self.w = np.random.rand(self.n_class, X_train.shape[1])

        for epoch in range(self.epochs):
            for x, y in zip(X_train, y_train):
                scores = np.dot(x, self.w.T)
                probs = self.softmax(scores)
                gradient = probs
                gradient[y] -= 1
                # Update weights using the gradient
                self.w -= lr * np.outer(gradient, x)
                self.w - + r * self.w
            y_pred_train = self.predict(X_train)
            accuracy = np.mean(y_pred_train == y_train) * 100
            accuracies.append(accuracy)
        return accuracies

    def train(self, X_train, y_train):
        self.model(X_train, y_train, self.lr, self.reg_const)

    def predict(self, X_test):
        scores = np.dot(X_test, self.w.T)
        probs = np.apply_along_axis(self.softmax, 1, scores)
        return np.argmax(probs, axis=1)
    # plotting epoch

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

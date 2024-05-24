import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n_class, lr, epochs):
        self.weights = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def predict(self, X) -> np.ndarray:
        scores = np.dot(X, self.weights.T)
        return np.argmax(scores, axis=1)

    def model(self, X_train, y_train, lr):
        accuracies = []
        self.weights = np.zeros((self.n_class, X_train.shape[1]))
        for i in range(self.epochs):
            for x, y in zip(X_train, y_train):
                y_pred = self.predict(x.reshape(1, -1))
                if y_pred != y:
                    self.weights[y] += lr * x
                    self.weights[y_pred] -= lr * x
            y_pred_train = self.predict(X_train)
            accuracy = np.mean(y_pred_train == y_train) * 100
            accuracies.append(accuracy)
            # print(f'Epoch {i + 1}th iteration, Training Accuracy: {accuracy}%')
        return accuracies

    def train(self, X_train, y_train):
        self.model(X_train, y_train, self.lr)

    def plot_epoches(self, X_train, y_train):
        accuracy_epochs = self.model(
            X_train, y_train, self.lr)

        plt.plot(range(1, self.epochs+1), accuracy_epochs, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.grid(True)
        plt.show()

    def plot_lr(self, X_train, y_train):
        accuracy_lr = []
        for lr in self.lr:
            accuracies = self.model(X_train, y_train, lr)
            accuracy_lr.append(accuracies[-1])
        plt.plot(self.lr, accuracy_lr, marker='o')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Learning Rate')
        plt.grid(True)
        plt.show()
        #         accuracies.append((epochs, lr, accuracy))
        # epochs_values = [item[0] for item in accuracies]
        # learning_rate_values = [item[1] for item in accuracies]
        # accuracy_values = [item[2] for item in accuracies]

        # max_accuracy = max(accuracy_values)

        # max_accuracy_index = accuracy_values.index(max_accuracy)

        # best_epochs = epochs_values[max_accuracy_index]
        # best_lr = learning_rate_values[max_accuracy_index]

        # print("Epochs with maximum accuracy:", best_epochs)
        # print("Learning rate with maximum accuracy:", best_lr)

        # fig = plt.figure(figsize=(10, 6))

        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(epochs_values, learning_rate_values,
        #            accuracy_values, c='blue', marker='o')
        # ax.set_xlabel('Epochs')
        # ax.set_ylabel('Learning Rate')
        # ax.set_zlabel('Accuracy')
        # plt.title('Accuracy vs Epochs and Learning Rates')
        # plt.show()

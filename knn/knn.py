import torch


class KNN():
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, T):
        T_squared = torch.sum(T ** 2, axis=1).reshape(-1, 1)
        X_train_squared = torch.sum(self.X_train ** 2, axis=1).reshape(1, -1)
        dist = T_squared + X_train_squared - 2 * T @ self.X_train.T  # (i,)+(,j)-(i, j)
        k_nearest_labels = self.y_train[torch.argsort(dist, dim=1)[:, :K]]

        y_pred = torch.zeros(T.shape[0])
        for row in range(k_nearest_labels.shape[0]):
            y_pred[row] = torch.bincount(k_nearest_labels[row]).argmax()

        return y_pred


if __name__ == "__main__":
    torch.manual_seed(0)

    d = 20  # 特征维度
    n = 100  # 训练样本数量
    C = 10  # 类别的数量
    t = 40  # 测试样本数量
    K = 7  # 邻居数量

    X_train = torch.rand(n, d)
    y_train = torch.randint(1, C, (n,))
    T = torch.rand(t, d)

    knn = KNN()
    knn.train(X_train, y_train)
    predicted = knn.predict(T)
    print(predicted.shape, predicted)

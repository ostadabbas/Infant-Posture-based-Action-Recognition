class StructuredHingeLoss:
    def __init__(self, y, y_hat, eps=0):
        self.y = y
        self.y_hat = y_hat
        self.eps = eps

    def calculate_loss(self):
        loss = 0

        for i in range(len(self.y)):
            loss += max(0, abs(self.y[i] - self.y_hat[i]) - self.eps)

        return loss / len(self.y)
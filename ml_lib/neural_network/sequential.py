from ml_lib.neural_network.get_optimizers import get_optimizer
class Sequential:
    def __init__(self, layers=None):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self.layers.append(layer)

    
    def compile(self, optimizer='adam', loss=None,lr=0.001):
        self.optimizer = get_optimizer(optimizer,lr)
        self.loss_fn = loss

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

   

    def fit(self, X, y, epochs=100, verbose=20):
        if self.optimizer is None or self.loss_fn is None:
            raise ValueError("Call compile() before fit()")

        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)

            # Loss
            loss_value = self.loss_fn.forward(y_pred, y)

            # Backward
            grad = self.loss_fn.backward()
            self.backward(grad)

            # Update using optimizer
            for layer in self.layers:
                if hasattr(layer, "W"):  # layers with params
                    self.optimizer.update(layer)

            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch} | Loss: {loss_value:.4f}")

    def predict(self, X):
        return self.forward(X)

import numpy as np
from .get_optimizers import get_optimizer
class Model:
    def __init__(self, shared_layers, classification_head, regression_head):
        self.shared_layers = shared_layers
        self.classification_head = classification_head
        self.regression_head = regression_head
        
        self.losses = None
        self.loss_weights = None
        self.optimizer = None

    def compile(self, optimizer, losses, loss_weights=None,lr=0.001):
        self.optimizer = get_optimizer(optimizer,lr=0.001)
        self.losses = losses
        self.loss_weights = loss_weights or [1.0, 1.0]

    def _forward(self, X):
        shared_out = self.shared_layers.forward(X)

        cls_out = self.classification_head.forward(shared_out)
        reg_out = self.regression_head.forward(shared_out)

        return cls_out, reg_out, shared_out

    def _backward(self, grad_cls, grad_reg):
        # Ensure shapes are correct
        if grad_reg.ndim == 1:
            grad_reg = grad_reg.reshape(-1, 1)

        grad_cls = self.classification_head.backward(grad_cls)
        grad_reg = self.regression_head.backward(grad_reg)

        total_grad = grad_cls + grad_reg

        self.shared_layers.backward(total_grad)

    def fit(self, X, y, epochs=50, verbose=True):
        Y_cls, Y_reg = y

        for epoch in range(epochs):
            cls_pred, reg_pred, _ = self._forward(X)

            # Compute losses
            cls_loss = self.losses[0].forward(cls_pred, Y_cls)
            reg_loss = self.losses[1].forward(reg_pred, Y_reg)

            # Weighted combined loss (just informative)
            total_loss = (self.loss_weights[0] * cls_loss +
                          self.loss_weights[1] * reg_loss)

            # Gradients
            grad_cls = self.losses[0].backward() * self.loss_weights[0]
            grad_reg = self.losses[1].backward() * self.loss_weights[1]

            self._backward(grad_cls, grad_reg)

            # Update all trainable layers
            for layer in self.layers():
                if hasattr(layer, "W"):
                    self.optimizer.update(layer)

            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Cls Loss: {cls_loss:.4f} | Reg Loss: {reg_loss:.4f}")

    def predict(self, X):
        cls_pred, reg_pred, _ = self._forward(X)
        return cls_pred, reg_pred

    def layers(self):
        return (
            self.shared_layers.layers +
            self.classification_head.layers +
            self.regression_head.layers
        )


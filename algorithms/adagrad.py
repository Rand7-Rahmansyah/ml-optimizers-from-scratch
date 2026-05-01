import numpy as np
import matplotlib.pyplot as plt

LOSS_TYPES = {"mse", "cross_entropy"}

class Adagrad:
    def __init__(
        self,
        *,
        loss="mse",
        lr=0.01,
        epochs=100,
        batch_size=32,
        epsilon=0.5,
        shuffle=True,
        random_state=None,
        verbose=1
    ):
  

"""
_losses.py
----------

Callable loss function classes untuk AdagradClassifier.

Desain:
  - Setiap loss adalah callable class dengan __call__(y_true, y_pred) -> float
  - Gradient TIDAK dihitung disini - ada di activations.py (separation of concern)
  - Semua class mewarisi BeseLoss untuk kontrak yang konsisten

Penggunaan di _model.py:
     loss_fn = get_loss("mse")
     loss_val = loss_fn(y_true, y_pred)  # scalar

Tersedia:
   - MSELoss                : Mean Squared Error  -> 0.5 * mean((y - ŷ)²)
   - CrossEntropyLoss       : Binary Cross-Entropy -> -mean(y * log(ŷ))
   - get_loss()             : factory function     -> loss by name string
"""

from __feature__ import annotations


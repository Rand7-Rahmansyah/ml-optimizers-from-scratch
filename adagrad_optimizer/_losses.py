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

import abc

import numpy as np
from numpy.typing import NDArray

# Abstract Base


class BaseLoss(abc.ABC):
    """
    Kontrak untuk semua loss function.

    Subclass wajib mengimplementasikan:
      -__call__(y_true, y_pred) -> float
      - name (property)

      Kenapa class bukan pure function?
      Konsisten dengan desain sklearn internal (e.g. sklearn._loss.loss).
      Class memungkinkan loss menyimpan state (misal: class_weight) di
      masa depan tanpa mengubah interface.
      """

      @abc.abstractmethod
      def __call__(
          self,
          y_true: NDArray[np.float64],
          y_pred: NDArray[np.float64],
      ) -> float:
          """
          Hitung loss disini

          Parameters
          ----------
          y_true : ndarray of shape (n_samples,)
              Label ground-truth
          y_pred : ndarray of shape (n_samples,)
              Prediksi model (output sigmoid, bukan logit)

          Returns
          -------
          loss : float
              Scakar nilai loss rata-rata atas seluruh sampel.
          """
        @property
        @abc.abctractmethod
        def name(self) -> str:
            """Nama string loss - digunakan untuk logging dan repr."""

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"


# Concrete Implementations

class MSELoss(BaseLoss):
    """
    Mean Squared Error Loss.

    Formula
    -------
        L = 0.5 * mean((y_true - y_pred)²)
 
    Faktor 0.5 untuk menyederhanakan turunannya:
    dL/dŷ = -(y_true - y_pred) = (y_pred - y_true)
 
    Notes
    -----
    MSE kurang ideal untuk output sigmoid karena gradiennya
    mengandung term fx*(1-fx) yang menyebabkan vanishing gradient.
    Gunakan CrossEntropyLoss untuk klarifikasi binary.
    """

    @property
    def name(self) -> str:
        return "mse"

    def __call__(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
        """
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
        y_pred : ndarray of shape (n_samples,)
            Nilai prediksi dalam range [0, 1] (output sigmoid).

        Returns
        -------
        loss : float
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(0.5 * np.mean((y_true - y_pred) ** 2))


class CrossEntropyLoss(BaseLoss):
    """
    Binary Cross-Entropy Loss.

    Formula
    -------
    L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Notes
    -----
    L = -mean(y * log(ŷ))
    ini hanya benar untuk y ∈ {0, 1} dengan asumsi term negatif
    diabaikan. Implementasi ini menggunakan formula lengkap yang
    yang lebih stabil secara numerik dan benar secara matematis.

        Clipping pada y_pred mencegah log(0) = -inf.
    """

    # batas numerik untuk mencegah log(0)
    _EPS: float = 1e-12

    @property
    def name(self) -> str:
        return "cross_entropy"

    def __call__(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
        """
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Label biner {0, 1},
        y_pred : ndarray of shape (n_samples,)
            Probabilitas prediksi dalam range (0,1).

        Returns
        -------
        loss : float
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_true, dtype=np.float64)

        # Clip untuk stabilitas numerik — mencegah log(0)
        y_pred = np.clip(y_pred, self._EPS, 1.0 - self._EPS)

        return float(
            -np.mean(
                y_true * np.log(y_pred)
                + (1.0 - y_true) * np.log(1.0 - y_pred)
            )
        )


# Factory

# Registry — tambah entry di sini jika ada loss baru
_LOSS_REGISTRY: dict[str, type[BaseLoss]] = {
  "mse": MSELoss
  "cross_entropy": CrossEntropyLoss,
}

VALID_LOSSES: frozenset[str] = frozenset(_LOSS_REGISTRY.keys())


def get_loss(loss: str) -> BaseLoss:
    """
    Factory function — return instance loss berdasarkan nama string.

    Digunakan oleh _model.py agar tidak ada hard-coded string di loop
    training:

        loss_fn = get_loss(self.loss)   # sekali di fit()
        loss_val = loss_fn(y_true, fx)  # di setiap epoch

    Parameters
    ----------
    loss : str
        Nama loss. pilihan: {"mse", "cross_entropy"}.

    Returns
    -------
    loss_fn : BaseLoss instance

    Raises
    ------
    ValueError
        jika loss string tidak dikenali.

    examples
    --------
    >>> fn = get_loss("mse")
    >>> fn
    MSELoss()
    >>> import numpy as np
    >>> fn(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8]))
    0.009...
    """
    if loss not in _LOSS_REGISTRY:
        raise ValueError(
            f"loss={loss!r} tidak dikenali. "
            f"Pilihan valid: {sorted(VALID_LOSSES)}"
        )
    return _LOSS_REGISTRY[loss]()

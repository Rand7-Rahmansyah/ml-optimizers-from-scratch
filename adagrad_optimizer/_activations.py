"""
_activations.py
---------------
Callable activation function classes untuk AdagradClassifier.

Desain:
  - Konsisten dengan _losses.py — setiap activation adalah callable class
  - __call__()   -> alias forward(), untuk interface yang natural
  - forward()    -> komputasi aktivasi (forward pass)
  - gradient()   -> komputasi gradient (backward pass)
  - Semua class mewarisi BaseActivation untuk kontrak yang konsisten

Kenapa forward() dan gradient() dipisah?
  - _model.py membutuhkan keduanya di titik berbeda dalam loop:
      fx  = activation.forward(z)       # forward pass
      grad = activation.gradient(fx)    # backward pass (input = output forward)
  - Memisahkan keduanya menghindari komputasi ulang sigmoid yang mahal

Penggunaan di _model.py:
    act = get_activation("sigmoid")
    fx   = act.forward(z)        # atau act(z)
    grad = act.gradient(fx)      # gradient terhadap input z

Tersedia:
  - Sigmoid           : σ(z) = 1 / (1 + e⁻ᶻ)
  - get_activation()  : factory function → activation by name string
"""

from __future__ import annotations

import abc

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Abstract Base
# ======================================================================

class BaseActivation(abc.ABC):
    """
    Kontrak untuk semua activation function.

    Subclass wajib mengimplementasikan:
      - forward(z)   -> ndarray
      - gradient(fx) -> ndarray
      - name (property)

    __call__ di-delegate ke forward() sehingga kedua style ini valid:
        act.forward(z)   # eksplisit
        act(z)           # shorthand
    """

    @abc.abstractmethod
    def forward(
        self,
        z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Hitung output aktivasi dari pre-activation z.

        Parameters
        ----------
        z : ndarray of shape (n_samples,)
            Pre-activation: z = w·x + b

        Returns
        -------
        fx : ndarray of shape (n_samples,)
            Output aktivasi dalam range sesuai fungsi.
        """

    @abc.abstractmethod
    def gradient(
        self,
        fx: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Hitung gradient dσ/dz dari output aktivasi fx.

        Menerima OUTPUT forward (bukan input z) sebagai argumen
        karena sebagian besar aktivasi bisa dinyatakan dalam
        output-nya sendiri — menghindari komputasi ulang.

        Contoh sigmoid:
            σ'(z) = σ(z) * (1 - σ(z)) = fx * (1 - fx)

        Parameters
        ----------
        fx : ndarray of shape (n_samples,)
            Output dari forward() — BUKAN z mentah.

        Returns
        -------
        grad : ndarray of shape (n_samples,)
            Gradient elemen per elemen dσ/dz.
        """

    def __call__(
        self,
        z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Shorthand untuk forward(z)."""
        return self.forward(z)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Nama string aktivasi — digunakan untuk logging dan repr."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ======================================================================
# Concrete Implementations
# ======================================================================

class Sigmoid(BaseActivation):
    """
    Sigmoid Activation Function.

    Forward
    -------
    σ(z) = 1 / (1 + e⁻ᶻ)

    Output range: (0, 1) — cocok untuk klasifikasi binary.

    Gradient
    --------
    σ'(z) = σ(z) * (1 - σ(z))
           = fx * (1 - fx)

    Digunakan di backward pass untuk kedua loss:

    MSE gradient:
        ∂L/∂w = (fx - y) * fx * (1 - fx) * x
        ∂L/∂b = (fx - y) * fx * (1 - fx)
        → term fx*(1-fx) berasal dari sigmoid.gradient(fx)

    CrossEntropy gradient (lebih bersih):
        ∂L/∂w = (fx - y) * x
        ∂L/∂b = (fx - y)
        → sigmoid.gradient() ter-cancel dengan cross-entropy derivative
          sehingga tidak dibutuhkan, tapi tetap tersedia untuk konsistensi

    Notes
    -----
    Numerically stable: np.exp(-z) aman untuk z dalam range float64.
    Untuk z sangat negatif (~-709), exp(-z) → inf tapi 1/(1+inf) → 0
    yang masih valid. np.clip tidak dibutuhkan di forward().
    """

    @property
    def name(self) -> str:
        return "sigmoid"

    def forward(
        self,
        z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        z : ndarray of shape (n_samples,) atau scalar
            Pre-activation value z = w·x + b

        Returns
        -------
        fx : ndarray of shape (n_samples,)
            σ(z) dalam range (0, 1).
        """
        z = np.asarray(z, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-z))

    def gradient(
        self,
        fx: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        fx : ndarray of shape (n_samples,)
            Output dari forward() — nilai σ(z).
            PENTING: bukan z mentah, tapi hasil sigmoid.

        Returns
        -------
        grad : ndarray of shape (n_samples,)
            σ'(z) = fx * (1 - fx)

        Examples
        --------
        >>> act = Sigmoid()
        >>> fx = act.forward(np.array([0.0]))
        >>> fx
        array([0.5])
        >>> act.gradient(fx)
        array([0.25])   # 0.5 * (1 - 0.5) = 0.25
        """
        fx = np.asarray(fx, dtype=np.float64)
        return fx * (1.0 - fx)


# ======================================================================
# Factory
# ======================================================================

# Registry — tambah entry di sini jika ada aktivasi baru (ReLU, Tanh, dll)
_ACTIVATION_REGISTRY: dict[str, type[BaseActivation]] = {
    "sigmoid": Sigmoid,
}

VALID_ACTIVATIONS: frozenset[str] = frozenset(_ACTIVATION_REGISTRY.keys())


def get_activation(activation: str) -> BaseActivation:
    """
    Factory function — return instance activation berdasarkan nama string.

    Digunakan oleh _model.py sekali di awal fit():

        act = get_activation("sigmoid")
        fx   = act.forward(z)      # forward pass
        grad = act.gradient(fx)    # backward pass

    Parameters
    ----------
    activation : str
        Nama aktivasi. Saat ini hanya: {"sigmoid"}.

    Returns
    -------
    activation_fn : BaseActivation instance

    Raises
    ------
    ValueError
        Jika activation string tidak dikenali.

    Examples
    --------
    >>> act = get_activation("sigmoid")
    >>> act
    Sigmoid()
    >>> act(np.array([0.0]))
    array([0.5])
    """
    if activation not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"activation={activation!r} tidak dikenali. "
            f"Pilihan valid: {sorted(VALID_ACTIVATIONS)}"
        )
    return _ACTIVATION_REGISTRY[activation]()

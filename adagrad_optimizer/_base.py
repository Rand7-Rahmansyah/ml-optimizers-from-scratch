"""
Abstract base class untuk AdagradClassifier

kontrak sklearn yang di penuhi:
   - BaseEstimator    : get_params() / set_params() / __repr__() otomatis
   - ClassifierMixin  : score() via accuracy_score otomatis

Subclass WAJIB mengimplementasikan:
   - fit(X, y)              -> self
   - predict(X)             -> ndarray shape (n_samples,)
   - predict_proba(X)       -> ndarray shape (n_samples, n_classes)

Subclass DILARANG override:
   - get_params()
   - set_params()
   - score()           (kecuali ada alasan yang kuat)
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class BaseAdagradClassifier(BaseEstimator, ClassifierMixin, metaclass=abc.ABCMeta):
    """
    Abstract base untuk semua Adagrad-based classifier.

    Mewarisi dari BaseEstimator dan ClassifierMixin sehingga:

    * ``get_params()`` / ``set_params()`` tersedia otomatis - dibutuhkan
      oleh GridsearchCV, Pipeline, dan clone().
    * ``score(X, y)`` tersedia otomatis - mengembalikan accuracy_score.

    parameters
    ----------
    loss : {"mse", "cross_entropy"}, default="cross_entropy"
        Fungsi loss yang digunakan selama training.
    lr : float, default=0.01
        Learning rate awal (η). Harus > 0.
    epoch : int, default=100
        jumlah iterasi penuh atas seluruh dataset.
    batch_size : int, default=32
        ukuran mini-batch. harus > 0 dan ≤ n_samples.
    epsilon : float, default=1e-8
        konstanta numerik kecil untuk stabilitas pembagian pada
        Adagrad update. Nilai konvensional: 1e-7 atau 1e-8.
    shuffle : bool, default=True
        Apakah data di-shuffle setiap epoch
    random_state : int or None, default=None
        seed untuk reprodusibilitas.
    verbose : int, default=0
        Level logging. 0 = diam, 1 = loss per epoch.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Label kelas unik. Di-set saat fit()
    n_features_in_ : int
        jumlah fitur input. Di-set saat fit()
        dibutuhkan oleh sklearn untuk validasi predict().
    is_fitted_ : bool
        True setelah fit() berhasil dipanggil.
    losss_history_ : list of float
        Loss di setiap akhir epoch.

    Notes
    -----
    Konvensi penamaan sklearn (wajib diikuti subclass):

    * parameter konstruktor  -> nama tanpa trailing underscore
    * Attributes hasil fit() -> nama dengan trailing underscore (``w_``, ``b_``)
    * Private helpers        -> nama dengan leading underscore (``_sigmoid``)
    """

    # Daftar loss yang valid — subclass bisa override jika perlu
    _VALID_LOSSES: frozenset[str] = frozenset({"mse", "cross_entropy"})

   def __init__(
      self,
      *,
      loss: str = "cross_entropy",
      lr: float = 0.01,
      epochs: int = 100,
      batch_size: int = 32,
      epsilon: float = 1e-8,
      shuffle: bool = True,
      random_state: Optional[int] = None,
      verbose: int = 0,
   ) -> None:
      # PENTING: BaseEstimator.get_params() membaca atribut ini
      # via inspect.signature - nama parameter HARUS sama persis
      # dengan nama atribut instance. Jangan lakukan self.lr = lr * 2
      # di sini; transformasi di lakukan di fit()
      self.loss = loss
      self.lr = lr
      self.epochs = epochs
      self.batch_size = batch_size
      self.epsilon = epsilon
      self.shuffle = shuffle
      self.random_state = random_state
      self.verbose = verbose
      
   # Abstrack interface - wajib diimplementasikan oleh subclass

   @abc.abstractmethod
   def fit(self, X: ArrayLike, y: ArrayLike) -> "BaseAdagradClassifier":
       """
       Latih model pada data(X, y).

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)
       y : array-like of shape (n_samples)

       Returns
       -------
       self : object
           Instance yang sudah difit (untuk method chaining).
       """

    @abc.abstractmethod
    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """
        Prediksi label kelas untuk X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """

    @abc.abstractmethod
    def predit_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Probabilitas kelas untuk setiap sampel di X

        Parameters
        ----------
        X : array-like of shape (n_sampless, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Baris ke-i adalah probabilitas untuk sampel ke-i
            setiap baris menjumlahkan ke 1.0
        """

    # Concrete helpers - tersedia untuk semua subclass

    def _validate_params(self) -> None:
        """
        Validasi hyperparameter sebelum training dimulai

        Dipanggil di awal fit() oleh subclass via super()._validate_params()
        atau secara eksplisit.

        Raises
        ------
        ValueError
            Jika ada parameter yang tidak valid.
        """
        if self.loss not in self._VALID_LOSSES:
            raise ValueError(
                f"loss={self.loss!r} tidak valid."
                f"Pilihan: {sorted(self._VALID_LOSSES)}"
            )
         if self.lr <= 0:
             raise ValueError(f"lr harus > 0, dapat: {self.lr}")
         if self.epochs <= 0:
             raise ValueError(f"epochs harus > 0, dapat: {self.epochs}")
         if self.batch_size <= 0:
             raise ValueError(f"batch_size harus > 0, dapat: {self.batch_size}")
         if self.epsilon <= 0:
             raise ValueError(f"epsilon harus > 0, dapat: {self.epsilon}")

    def _check_is_fitted(self) -> None:
        """
        Pastikan fit() sudah dipanggil sebelum predicts/predict_proba

        Raises
        ------
        sklearn.exceptions.NotFittedError
            Jika model belum difit.
        """
        check_is_fitted(self, attributes=["classes_", "n_features_in_"])

    def _log(self, epoch: int, loss_val: float) -> None:
        """
        Print loss per epoch jika verbose >0.

        Parameters
        ----------
        epoch :int
        loss_val :float
        """
        if self.verbose > 0:
            print(f"Epoch {epoch:>4d} | loss = {loss_val:.6f}")
            
        
        
       





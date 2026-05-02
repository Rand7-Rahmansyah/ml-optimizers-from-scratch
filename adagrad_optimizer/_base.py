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


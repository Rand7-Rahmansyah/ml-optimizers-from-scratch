"""
_activations.py
---------------
Callable activation function classes untuk AdagradClassifier.

Desain:
  - konsisten dengan _losses.py - setiap activation adalah callable class
  - __call__()    -> alias forward(), untuk interface yang natural
  - forward()     -> komputasi aktivasi (forward pass)

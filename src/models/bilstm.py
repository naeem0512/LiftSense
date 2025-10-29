from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, optimizers

from ..persistence import ensure_dir


@dataclass
class BiLSTMModel:
    model: keras.Model

    @classmethod
    def create(cls, input_shape: Tuple[int, int], lr: float) -> "BiLSTMModel":
        inputs = layers.Input(shape=input_shape)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(3, activation="softmax")(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return cls(model=model)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        patience: int,
        output_dir: Path,
    ) -> Dict[str, Any]:
        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ]
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks_list,
        )

        plot_path = output_dir / "plots" / "training_curves.png"
        ensure_dir(plot_path.parent)
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Bi-LSTM Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return {"history": history.history, "plot_path": str(plot_path)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.model.predict(X, verbose=0), axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        self.model.save(path)

    @classmethod
    def load(cls, path: Path) -> "BiLSTMModel":
        model = keras.models.load_model(path)
        return cls(model=model)

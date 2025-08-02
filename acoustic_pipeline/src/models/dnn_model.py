import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, strip_pruning

class DNNClassifier:
    def __init__(self, input_shape, num_classes=2, prune=False, pruning_params=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.prune = prune
        self.pruning_params = pruning_params if pruning_params else {}
        self.model = self._build_model()

    def _maybe_prune(self, layer_class, *args, **kwargs):
        layer = layer_class(*args, **kwargs)
        return prune_low_magnitude(layer, **self.pruning_params) if self.prune else layer

    def _build_model(self):
        print("✨ Building DNN model...")
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(self._maybe_prune(Dense, 128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(self._maybe_prune(Dense, 64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(self._maybe_prune(Dense, 1, activation='sigmoid'))  # Binary classification
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50, batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        return history

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X).flatten()

    def save(self, path, strip=False):
        model_to_save = strip_pruning(self.model) if strip and self.prune else self.model
        model_to_save.save(path)

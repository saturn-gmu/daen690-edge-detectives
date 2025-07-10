# dnn_model.py
# Provides a simplified DNN architecture for binary classification.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_consistent_dnn_model(input_shape):
    """
    Builds a consistent DNN model for classification using the given input shape.
    
    Parameters:
    - input_shape: tuple, e.g. (64,) representing the number of input features
    
    Returns:
    - Compiled Keras model
    """
    inputs = Input(shape=input_shape, name="input")
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_dnn_model(X_train, y_train, X_val, y_val, input_shape=None):
    """Trains the simplified DNN model with early stopping."""
    if input_shape is None:
        input_shape = (X_train.shape[1],)
    elif isinstance(input_shape, int):
        input_shape = (input_shape,)

    model = build_consistent_dnn_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


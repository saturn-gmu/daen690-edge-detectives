# dnn_model.py
# This module defines and trains a deep neural network (DNN) classifier,
# and extracts bottleneck features for hybrid modeling.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision
from tensorflow.keras.utils import to_categorical
import numpy as np

# Build and train a simple fully connected DNN model
def train_dnn_model(X_train, y_train, X_val, y_val):
    # Define the model architecture
    input_layer = Input(shape=(X_train.shape[1],))

    # First dense hidden layer with L2 regularization, batch norm, and dropout
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Bottleneck layer (used later for feature extraction)
    bottleneck = Dense(64, activation='relu', name='bottleneck', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(bottleneck)

    # Output layer: 2-class softmax
    output = Dense(2, activation='softmax')(x)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision')])

    # Use early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, to_categorical(y_train),
        validation_data=(X_val, to_categorical(y_val)),
        epochs=15,
        batch_size=32,
        callbacks=[early_stop]
    )

    return model, history

# Extract bottleneck features from the trained DNN
def extract_bottleneck_features(model, X_train, X_test, scaler, df):
    # Define a new model that outputs from the bottleneck layer
    bottleneck_model = Model(inputs=model.input, outputs=model.get_layer("bottleneck").output)

    # Transform training and test sets
    X_train_b = bottleneck_model.predict(X_train)
    X_test_b = bottleneck_model.predict(X_test)

    # Transform all feature vectors for full-dataset RF evaluation
    X_all_b = bottleneck_model.predict(scaler.transform(np.vstack(df['features'].values)))

    return X_train_b, X_test_b, X_all_b
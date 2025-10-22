from typing import Optional
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    h = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding='causal', activation='relu')(x)
    h = Dropout(dropout_rate)(h)
    h = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding='causal', activation='relu')(h)
    h = Dropout(dropout_rate)(h)
    if x.shape[-1] != filters:
        x = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
    return Add()([x, h])

def build_tcn(hp, input_shape):
    filters     = hp.Choice('filters', [32, 48, 64, 96])
    blocks      = hp.Int('blocks', 3, 6, 1)
    kernel_size = hp.Choice('kernel_size', [2, 3, 5])
    dropout     = hp.Float('dropout', 0.1, 0.5, step=0.1)
    lr          = hp.Float('lr', 1e-4, 3e-3, sampling='log')
    batch_size  = hp.Choice('batch_size', [32, 64, 128])
    inp = Input(shape=input_shape)
    x = inp
    for i in range(blocks):
        x = residual_block(x, filters, kernel_size, dilation_rate=2**i, dropout_rate=dropout)
    x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    model._tune_batch_size = batch_size
    return model

def tune_and_train(X_tr, y_tr, X_val, y_val, max_epochs=60, directory='tcn_tuning', project_name='cmapss_tcn'):
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_tcn(hp, X_tr.shape[1:]),
        objective='val_loss', max_epochs=max_epochs, factor=3,
        directory=directory, project_name=project_name,
    )
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_tr, y_tr, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=[es], verbose=1)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_tcn(best_hp, X_tr.shape[1:])
    bs = getattr(model, "_tune_batch_size", 64)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=max_epochs, batch_size=bs, callbacks=[es], verbose=1)
    return model, history, best_hp, bs

def train_default(X_tr, y_tr, X_val, y_val, max_epochs=60):
    class DummyHP:
        def Choice(self, name, vals): return vals[0]
        def Int(self, n, a, b): return a
        def Float(self, n, a, b, step=None, sampling=None): return a
    model = build_tcn(DummyHP(), X_tr.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=max_epochs, batch_size=64, callbacks=[es], verbose=1)
    return model, hist
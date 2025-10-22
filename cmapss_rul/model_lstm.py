import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_model(hp, input_shape):
    units       = hp.Choice('units', [32, 64, 96, 128])
    layers      = hp.Int('layers', 1, 3, 1)
    bidir       = hp.Boolean('bidirectional', default=True)
    dropout     = hp.Float('dropout', 0.1, 0.5, step=0.1)
    lr          = hp.Float('lr', 1e-4, 3e-3, sampling='log')
    batch_size  = hp.Choice('batch_size', [32, 64, 128])

    x = inp = Input(shape=input_shape)
    for i in range(layers):
        return_sequences = (i < layers - 1)
        lstm = LSTM(units, return_sequences=True, activation='tanh')
        x = (Bidirectional(lstm)(x) if bidir else lstm(x))
        x = Dropout(dropout)(x)
        if not return_sequences:
            # convert sequence to vector via pooling
            x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    model._tune_batch_size = batch_size
    return model

def tune_and_train(X_tr, y_tr, X_val, y_val, max_epochs=60, directory='tuning', project_name='lstm'):
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, X_tr.shape[1:]),
        objective='val_loss', max_epochs=max_epochs, factor=3,
        directory=directory, project_name=project_name,
    )
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_tr, y_tr, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=[es], verbose=1)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_model(best_hp, X_tr.shape[1:])
    bs = getattr(model, "_tune_batch_size", 64)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=max_epochs, batch_size=bs, callbacks=[es], verbose=1)
    return model, hist, best_hp, bs

def train_default(X_tr, y_tr, X_val, y_val, max_epochs=60):
    from types import SimpleNamespace
    hp = SimpleNamespace(
        Choice=lambda name, vals: vals[0],
        Int=lambda n, a, b, step=None: a,
        Float=lambda n, a, b, step=None, sampling=None: a,
        Boolean=lambda name, default=False: default,
    )
    model = build_model(hp, X_tr.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=max_epochs, batch_size=64, callbacks=[es], verbose=1)
    return model, hist


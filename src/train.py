import logging
import glob
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt
from typing import Optional, Tuple

import model as m


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def create_training_dataset(filenames, start=0, num_files=128):
    all_notes = []

    for f in filenames[start:num_files]:
        notes = m.midi_to_notes(f)
        if notes is not None:
            all_notes.append(notes)

    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    key_order = ["pitch", "step", "duration"]
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    notes_ds.element_spec

    return (notes_ds, n_notes,)


def train(filenames,
          epochs=2,
          batch_size=64,
          seq_length=25,
          learning_rate=0.005,
          vocab_size=128):

    model = m.create_model(seq_length, learning_rate)
    model.summary()
    with open('./output/model.json', 'w') as f:
        f.write(model.to_json())

    batch = 50
    start = 0
    end = batch
    for _ in range(math.ceil(len(filenames) / batch)):
        logging.debug(f"start={start} num_files={end} {len(filenames)}")

        notes_ds, notes_n = create_training_dataset(
            filenames, start=start, num_files=end)
        seq_ds = m.create_sequences(notes_ds, seq_length, vocab_size)

        # the number of items in the dataset
        buffer_size = notes_n - seq_length
        train_ds = (seq_ds
                    .shuffle(buffer_size)
                    .batch(batch_size, drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        losses = model.evaluate(train_ds, return_dict=True)
        print(losses)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./training_checkpoints/checkpoint',
                monitor='loss',
                epoch=epochs,
                save_freq='epoch',
                save_weights_only=True,
                save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),
        ]
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size
        )

        plt.plot(history.epoch, history.history['loss'], label='total loss')
        plt.savefig(f'./output/loss{start}.png')

        start += batch
        end += batch

    return model


# Defining main function
def main():
    print(f"Tensorflow version: {tf.__version__}")

    data_dir = pathlib.Path('robbie-v1.0.0/clean/clean')
    filenames = glob.glob(str(data_dir/'*.mid*'))
    print('Number of files:', len(filenames))
    train(filenames)


if __name__ == "__main__":
    configure_logger()
    main()

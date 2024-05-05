import collections
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt
from typing import Optional, Tuple

import model as m


def create_training_dataset(filenames, start=0, num_files=128):
    all_notes = []

    for f in filenames[start:num_files]:
        notes = m.midi_to_notes(f)
        print(notes)
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


def train(filenames, seq_length=25, vocab_size=128):

    notes_ds, notes_n = create_training_dataset(
        filenames, num_files=len(filenames))
    seq_ds = m.create_sequences(notes_ds, seq_length, vocab_size)
    # seq_ds.element_spec

    # test_ds, test_n = create_training_dataset(filenames, 128, 256)
    # test_seq = create_sequences(notes_ds, seq_length, vocab_size)

    ###############

    for seq, target in seq_ds.take(1):
        print('sequence shape:', seq.shape)
        print('sequence elements (first 10):', seq[0: 10])
        print()
        print('target:', target)

    ###############

    # Batch the examples, and configure the dataset for performance.
    batch_size = 64
    buffer_size = notes_n - seq_length  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))
    # train_ds.element_spec

    ################

    model = m.create_model(seq_length)
    model.summary()

    losses = model.evaluate(train_ds, return_dict=True)
    print(losses)

    # model.evaluate(train_ds, return_dict=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]
    epochs = 10
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size
    )

    # buffer_size = test_n - seq_length  # the number of items in the dataset
    # test_ds = (test_seq
    #             .shuffle(buffer_size)
    #             .batch(batch_size, drop_remainder=True)
    #             .cache()
    #             .prefetch(tf.data.experimental.AUTOTUNE))
    # model.evaluate()

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.savefig('./output/loss.png')
    # plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    # plt.savefig('accuracy.png')
    # plt.show()

    with open('./output/model.mdl', 'w') as f:
        f.write(model.to_json())

    return model


# Defining main function
def main():
    # data_dir = pathlib.Path('data/maestro-v2.0.0')
    data_dir = pathlib.Path('data/robbie-v1.0.0/clean')
    # filenames = glob.glob(str(data_dir/'**/*.mid*'))
    filenames = glob.glob(str(data_dir/'*.mid*'))
    # data_dir = pathlib.Path('data/robbie-v1.0.0')
    # filenames = glob.glob(str(data_dir/'lmd_full/**/*.mid*'))
    print('Number of files:', len(filenames))
    train(filenames, seq_length=128)


if __name__ == "__main__":
    main()

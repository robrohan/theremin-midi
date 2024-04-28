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

    notes_ds, notes_n = create_training_dataset(filenames)
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
    epochs = 50
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
    plt.savefig('loss.png')
    # plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    # plt.savefig('accuracy.png')
    # plt.show()

    with open('./output/model.mdl', 'w') as f:
        f.write(model.to_json())

    return model


# Defining main function
def main():
    # data_dir = pathlib.Path('data/maestro-v2.0.0')
    # data_dir = pathlib.Path('data/robbie-v1.0.0')
    # download_dataset(data_dir)

    # filenames = glob.glob(str(data_dir/'lmd_full/**/*.mid*'))
    # print('Number of files:', len(filenames))

    # sample_file = filenames[0]
    # print(sample_file)

    # pm = pretty_midi.PrettyMIDI(sample_file)
    # print(pm)

    # # print(display_audio(pm))

    # # Just grab one instrument - should probably pick just one
    # # drums might be good ?
    # print('Number of instruments:', len(pm.instruments))
    # instrument = pm.instruments[0]
    # instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    # print('Instrument name:', instrument_name)

    # # Extract Notes
    # for i, note in enumerate(instrument.notes[:10]):
    #     note_name = pretty_midi.note_number_to_name(note.pitch)
    #     duration = note.end - note.start
    #     print(f'{i}: pitch={note.pitch}, note_name={note_name},' f' duration={duration:.4f}')

    # raw_notes = midi_to_notes(sample_file)
    # # raw_notes.head()
    # print(raw_notes)

    # get_note_names = np.vectorize(pretty_midi.note_number_to_name)
    # sample_note_names = get_note_names(raw_notes['pitch'])
    # print(sample_note_names[:10])

    # plot_piano_roll(raw_notes, count=100)
    # plot_piano_roll(raw_notes)

    # plot_distributions(raw_notes)

    # example_file = './output/example.mid'
    # example_pm = notes_to_midi(
    #     raw_notes,
    #     out_file=example_file,
    #     instrument_name=instrument_name)
    # display_audio(example_pm)

    ###############

    data_dir = pathlib.Path('data/maestro-v2.0.0')
    filenames = glob.glob(str(data_dir/'**/*.mid*'))
    # data_dir = pathlib.Path('data/robbie-v1.0.0')
    # filenames = glob.glob(str(data_dir/'lmd_full/**/*.mid*'))
    print('Number of files:', len(filenames))
    train(filenames)

    ###############

    # prompt_file = "prompt.mid"
    # raw_notes = midi_to_notes(prompt_file)
    # pm = pretty_midi.PrettyMIDI(prompt_file)
    # instrument = pm.instruments[0]
    # instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    # print('Instrument name:', instrument_name)

    # model = create_model()
    # model.load_weights('./training_checkpoints/ckpt_9')
    # print(model)
    # generated_notes = generate_notes(raw_notes, model=model, temperature=2.0)

    # # plot_piano_roll(generated_notes)
    # # plot_distributions(generated_notes)

    # out_file = './output/output.mid'
    # out_pm = notes_to_midi(
    #     generated_notes,
    #     out_file=out_file,
    #     instrument_name=instrument_name)


if __name__ == "__main__":
    main()

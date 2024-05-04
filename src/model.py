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

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000


def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    # return display.Audio(waveform_short, rate=_SAMPLING_RATE)
    return waveform_short


# def download_dataset(data_dir):
#     if not data_dir.exists():
#         tf.keras.utils.get_file(
#             'maestro-v2.0.0-midi.zip',
#             origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
#             extract=True,
#             cache_dir='.',
#             cache_subdir='data',
#         )


def midi_to_notes(midi_file: str, instrument_index=0) -> pd.DataFrame:
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[instrument_index]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start
    except Exception as e:
        print(f"could not load {midi_file} because {e}")
        return None

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = 'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))


def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0.0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size=128,
    key_order=["pitch", "step", "duration"],
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length+1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x/[vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


# def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
#     mse = (y_true - y_pred) ** 2
#     positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
#     return tf.reduce_mean(mse + positive_pressure)


def create_model(seq_length=25):
    input_shape = (seq_length, 3)
    learning_rate = 0.01

    inputs = keras.Input(input_shape)
    lstm = layers.LSTM(128)(inputs)
    l1 = layers.Dense(16)(lstm)
    l3 = layers.Dense(12)(l1)
    outputs = {
        'pitch': layers.Dense(128, name='pitch')(l3),
        'step': layers.Dense(1, name='step')(l3),
        'duration': layers.Dense(1, name='duration')(l3),
    }
    model = keras.Model(inputs, outputs)

    loss = {
        'pitch': keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': keras.losses.MeanSquaredError(),                 # mse_with_positive_pressure,
        'duration': keras.losses.MeanSquaredLogarithmicError(),  # mse_with_positive_pressure,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # model = tf.keras.Sequential([
    #     inputs,
    #     lstm,
    #     layers.Dense(7 * 7 * 128),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Reshape((7, 7, 128)),
    #     layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    #     outputs,
    # ])

    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 0.5,
            'step': 0.9,
            'duration': 0.8,
        },
        optimizer=optimizer,
    )
    # model.compile(loss=loss, optimizer=optimizer)
    return model


def predict_next_note(
        notes: np.ndarray,
        model: tf.keras.Model,
        temperature: float = 1.0) -> Tuple[int, float, float]:

    """Generates a note as a tuple of (pitch, step, duration),
        using a trained sequence model."""
    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def generate_notes(
        raw_notes,
        model: tf.keras.Model,
        seq_length=25,
        vocab_size=128,
        key_order=["pitch", "step", "duration"],
        temperature: float = 1.0):
    # temperature = 2.0
    num_predictions = 120

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0.0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(
            input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0),
                                axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    return generated_notes


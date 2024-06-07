import pretty_midi
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from typing import Any, Tuple

model_name = 'gpt2'


def encode_note(note: Any, prev_start: float) -> int:
    start = note.start
    end = note.end

    step = start - prev_start
    velocity = note.velocity
    pitch = note.pitch
    duration = end - start

    int_step = round(255 * step)
    int_duration = round(255 * duration)

    # move the step of the note up to the high bits
    encoded_note = int_step << 24
    # Next high bits duration
    encoded_note += int_duration << 16
    # upper part of first 16 is velocity
    encoded_note += velocity << 8
    # lower part of first 16 is pitch
    encoded_note += pitch

    return encoded_note


def decode_note(encoded_note: int, prev_start: float) -> Tuple:
    int_step = (encoded_note >> 24) & 255
    int_duration = (encoded_note >> 16) & 255
    velocity = (encoded_note >> 8) & 255
    pitch = (encoded_note) & 255

    # print(int_step / 255, int_duration / 255, velocity, pitch)
    f_step = (int_step / 255)*1000000/1000000
    f_duration = (int_duration / 255)*1000000/1000000

    start = float(prev_start + f_step)
    end = float(start + f_duration)

    return (velocity, pitch, start, end,)


def encode_midi(midi_file: str, instrument_index=0,
                window_size=64) -> pd.DataFrame:
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[instrument_index]
        # notes = collections.defaultdict(list)
        notes = []

        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        i = 0
        for note in sorted_notes:
            encoded_note = encode_note(note, prev_start)
            notes.append(encoded_note)
            prev_start = note.start
            i += 1
            if i >= window_size:
                break
    except Exception as e:
        print(f"could not load {midi_file} because {e}")
        return None

    return pd.DataFrame(notes)


def decode_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0.0
    for _, encoded_note in notes.iterrows():
        velocity, pitch, start, end = decode_note(encoded_note[0], prev_start)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def get_tokenizer():
    global model_name
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained('./models')

    return tokenizer


def create_model(seq_length=25, learning_rate=0.03):
    global model_name

    # Load the pre-trained GPT-2 model
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained('./models')

    # Define the input layer
    input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32,
                                      name="input_ids")

    # Check and print the vocabulary size
    vocab_size = model.config.vocab_size
    print(f"Vocabulary size: --> {vocab_size} <--")

    # Custom layer to ensure input IDs are within the valid range
    def valid_token_ids(input_ids):
        return tf.clip_by_value(input_ids, clip_value_min=0,
                                clip_value_max=vocab_size - 1)

    # Apply the validation layer
    valid_input_ids = tf.keras.layers.Lambda(valid_token_ids)(input_ids)

    # Get the model outputs
    outputs = model(valid_input_ids).logits

    # Define the final model
    model = tf.keras.Model(inputs=input_ids, outputs=outputs)

    # Compile the model with the specified loss and optimizer
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model


# def create_model(seq_length=25, learning_rate=0.03):
#     global model_name

#     model = TFGPT2LMHeadModel.from_pretrained(model_name)
#     model.save_pretrained('./models')

#     input_ids = tf.keras.layers.Input(shape=(seq_length,),
#                                       dtype=tf.int32,
#                                       name="input_ids")

#     vocab_size = model.config.vocab_size
#     print(f"Vocabulary size: --> {vocab_size} <--")

#     # Custom layer to ensure input IDs are within the valid range
#     def valid_token_ids(inputs):
#         input_ids = inputs['input_ids']
#         valid_ids = tf.clip_by_value(input_ids, clip_value_min=0,
#                                      clip_value_max=vocab_size - 1)
#         return valid_ids

#     # Use a Lambda layer to apply the custom validation function
#     valid_input_ids = tf.keras.layers.Lambda(
#         lambda inputs: valid_token_ids(inputs),
#         name="valid_input_ids")({'input_ids': input_ids})

#     # Pass the checked input IDs to the model
#     outputs = model(valid_input_ids).logits

#     model = tf.keras.Model(
#         inputs=input_ids,
#         outputs=outputs,
#         training=True,
#         name='gpt2_custom_model'
#     )

#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(
#         loss=loss,
#         optimizer=optimizer
#     )

#     return model

# # Example data
# import numpy as np

# # Assume you have a custom encoder that generates input IDs
# # For example:
# def custom_encoder(data):
#     # Replace this with your actual encoder logic
#     # Ensure the output is within the range [0, vocab_size - 1]
#     return np.random.randint(0, vocab_size, size=(len(data), seq_length))

# # Example training data
# train_data = ["example input 1", "example input 2"]  # Replace with your actual data
# train_labels = np.random.randint(0, vocab_size, size=(len(train_data), seq_length))  # Replace with actual labels

# # Encode the training data
# train_input_ids = custom_encoder(train_data)

# # Train the model
# model.fit(train_input_ids, train_labels, epochs=3, batch_size=2)
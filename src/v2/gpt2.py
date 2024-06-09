import collections
import fluidsynth
import glob
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

import model as m


class MIDITokenizer:
    def __init__(self,
                 pitch_range=(21, 108),
                 velocity_range=(0, 127),
                 duration_range=(0, 1000)):
        self.pitch_range = pitch_range
        self.velocity_range = velocity_range
        self.duration_range = duration_range

    def encode(self, midi_sequence):
        tokens = []
        for note in midi_sequence:
            pitch, duration, velocity = note
            tokens.append(f"{pitch}_{duration}_{velocity}")
        return " ".join(tokens)

    def decode(self, token_sequence):
        tokens = token_sequence.split()
        midi_sequence = []
        for token in tokens:
            pitch, duration, velocity = map(int, token.split('_'))
            midi_sequence.append((pitch, duration, velocity))
        return midi_sequence


# Prepare the dataset
class CustomDataset(tf.data.Dataset):
    def _generator(seq_length):
        with open('midi_data.txt', 'r') as f:
            for line in f:
                yield tokenizer.encode(line.strip(),
                                       return_tensors='tf',
                                       max_length=seq_length,
                                       truncation=True,
                                       padding='max_length')[0]

    def __new__(cls, seq_length=128):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=(seq_length,), dtype=tf.int32)
        )


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


# Initialize the model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# # Example MIDI sequence represented as text tokens
# midi_sequence = "60 64 67 72 60 64 67 72 60 64 67 72"

# # Encode the sequence
# input_ids = tokenizer.encode(midi_sequence, return_tensors='tf')

# # Generate music completion
# output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# # Decode the generated sequence
# generated_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_sequence)


# Example MIDI sequence
midi_sequence = [(60, 500, 64), (64, 500, 64), (67, 500, 64), (72, 500, 64)]

# Initialize the tokenizer
midi_tokenizer = MIDITokenizer()

# Encode the sequence
encoded_sequence = midi_tokenizer.encode(midi_sequence)
print(encoded_sequence)

# Decode the sequence
decoded_sequence = midi_tokenizer.decode(encoded_sequence)
print(decoded_sequence)

# Load the tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Prepare the dataset
# dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path='midi_data.txt',
#     block_size=128)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

seq_length = 128
dataset = CustomDataset(seq_length)
dataset = dataset.shuffle(1000).batch(4).prefetch(
    tf.data.experimental.AUTOTUNE
)

############################################################
# seq_length = 25
# vocab_size = 128
# print(f"Tensorflow version: {tf.__version__}")
# data_dir = pathlib.Path('data/robbie-v1.0.0/clean')
# filenames = glob.glob(str(data_dir/'*.mid*'))
# print('Number of files:', len(filenames))
# # train(filenames, seq_length=64)
# notes_ds, notes_n = create_training_dataset(filenames, num_files=len(filenames))
# seq_ds = m.create_sequences(notes_ds, seq_length, vocab_size)
############################################################


# Prepare data collator for language modeling
def data_collator(features):
    batch = tf.convert_to_tensor(features)
    return {'input_ids': batch, 'labels': batch}


input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32, name="input_ids")
outputs = model(input_ids).logits  # Use logits directly
model = tf.keras.Model(inputs=input_ids, outputs=outputs)

# loss = {
#     "pitch": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     "step": keras.losses.MeanSquaredError(),
#     "duration": keras.losses.MeanSquaredLogarithmicError(),
# }
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(
    loss=loss,
    # loss_weights={
    #     "pitch": 0.5,
    #     "step": 0.9,
    #     "duration": 0.8,
    # },
    # optimizer=optimizer,
    optimizer='adam'
)

# Compile the model
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(
#     optimizer=optimizer,
#     loss=loss
# )

# Fine-tune the model
model.fit(dataset, epochs=3)
# model.fit(seq_ds, epochs=3)

# Save the model
model.save_pretrained('./music_completion_model')
tokenizer.save_pretrained('./music_completion_model_token')


# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./output',
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
# )



# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=seq_ds,
#     # train_dataset=dataset,
# )

# # Fine-tune the model
# trainer.train()

# # Save the model
# trainer.save_model('./music_completion_model')

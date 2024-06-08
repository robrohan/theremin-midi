from model import notes_to_midi, midi_to_notes
from model2 import encode_midi, decode_midi, create_model, get_tokenizer

import numpy as np
import tensorflow as tf


def main():
    # test_file = './input/chords_75_Bb.midi'
    # test_file = './input/melody_75_F#.midi'
    # test_file = './input/notes.mid'
    # test_file = './input/prompt.mid'
    # test_file = "./output/output.mid"
    # test_file = "./input/test_long.midi"
    test_file = "./robbie-v1.0.0/clean/clean/0a5eefdc024a8076b0764636da85ae6f37b14cd8.midi"

    raw_notes = midi_to_notes(test_file)
    # print(raw_notes)
    notes_to_midi(
        raw_notes,
        out_file="./output/test1.midi",
        instrument_name=("Acoustic Grand Piano"),
        velocity=90,
    )

    raw_notes = encode_midi(test_file, 0, 2048)

    midi_chars = []
    for _, row in raw_notes.iterrows():
        try:
            midi_char = int(row[0].astype(np.uint32))
            byte_array = midi_char.to_bytes(4, 'big')
            unicode_character = byte_array.decode('utf-8')
            midi_chars.append(unicode_character)
        except Exception as e:
            print(e)
            print(f"{midi_char:32b} {byte_array}")
    midi_string = "".join(midi_chars)
    with open('output/midi_as_text.txt', 'w') as f:
        f.write(midi_string)

    decode_midi(
        raw_notes,
        out_file="./output/test2.midi",
        instrument_name=("Acoustic Grand Piano"),
    )

    # custom_input_ids = [i % vocab_size for i in custom_input_ids]

    # seq_length = 25
    #####################
    # model = create_model(seq_length=seq_length, learning_rate=0.03)
    # print(model)
    #####################

    #####################
    # tokenizer = get_tokenizer(model)
    # tokenizer.encode(raw_notes,
    #                  return_tensors='tf',
    #                  max_length=seq_length,
    #                  truncation=True,
    #                  padding='max_length')[0]
    #####################

    # input_ids_tensor = tf.constant([custom_input_ids], dtype=tf.int32)

    # dataset = tf.data.Dataset.from_tensor_slices(raw_notes)

    # dataset = tf.data.Dataset.from_generator(
    #     cls._generator,
    #     output_signature=tf.TensorSpec(shape=(seq_length,), dtype=tf.int32)
    # )

    #####################
    # model.fit(dataset, epochs=3)
    #####################


if __name__ == "__main__":
    main()

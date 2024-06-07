from model import notes_to_midi, midi_to_notes
from model2 import encode_midi, decode_midi, create_model, get_tokenizer

import tensorflow as tf


def main():
    # test_file = './input/chords_75_Bb.midi'
    # test_file = './input/prompt.mid'
    test_file = "./output/output.mid"

    raw_notes = midi_to_notes(test_file)
    print(raw_notes)
    notes_to_midi(
        raw_notes,
        out_file="./output/test1.midi",
        instrument_name=("Acoustic Grand Piano"),
        velocity=90
    )

    raw_notes = encode_midi(test_file, 0, 64)
    print(raw_notes)
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

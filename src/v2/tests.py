from midichar import encode_midi, decode_midi
import logging
import pathlib
import glob
import numpy as np
import tensorflow as tf


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def test_file():
    # test_file = './input/chords_75_Bb.midi'
    # test_file = './input/melody_75_F#.midi'
    # test_file = './input/notes.mid'
    # test_file = './input/prompt.mid'
    # test_file = "./output/output.mid"
    # test_file = "./input/test_long.midi"
    test_file = "./robbie-v1.0.0/clean/clean/0a5eefdc024a8076b0764636da85ae6f37b14cd8.midi"

    # raw_notes = midi_to_notes(test_file)
    # # print(raw_notes)
    # notes_to_midi(
    #     raw_notes,
    #     out_file="./output/test1.midi",
    #     instrument_name=("Acoustic Grand Piano"),
    #     velocity=90,
    # )

    seq_length = 64
    raw_notes = encode_midi(test_file, 0, seq_length)

    midi_chars = []
    for i, raw_note in enumerate(raw_notes):
        try:
            midi_char = int(raw_note.astype(np.uint32))
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


def main():
    print(f"Tensorflow version: {tf.__version__}")

    data_dir = pathlib.Path('robbie-v1.0.0/clean/clean')
    filenames = glob.glob(str(data_dir/'*.mid*'))
    print('Number of files:', len(filenames))

    test_file()


if __name__ == "__main__":
    configure_logger()
    main()

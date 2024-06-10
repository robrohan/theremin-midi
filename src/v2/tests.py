import logging
import pathlib
import glob
import numpy as np
import tensorflow as tf

from midichar import encode_midi, decode_midi, encoded_notes_to_str, str_to_encoded_notes

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

    seq_length = 64

    logging.info(f"encoding midi file with a length of {seq_length}")
    raw_notes = encode_midi(test_file, 0, seq_length)

    logging.info("encoding the raw notes to utf8")
    midi_string = encoded_notes_to_str(raw_notes)

    logging.info("saving to a text file as plain text")
    with open('output/midi_as_text.txt', 'w') as f:
        f.write(midi_string)

    #########################################################

    logging.info("reading plain text file back in")
    string_from_file = ""
    with open('output/midi_as_text.txt', 'r') as inf:
        string_from_file = inf.read()

    logging.info("converting utf8 chars back into integers")
    from_file = str_to_encoded_notes(string_from_file)

    logging.info("creating new midi file from text file notes")
    decode_midi(
        from_file,
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

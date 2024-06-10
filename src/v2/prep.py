import logging
import pathlib
import glob
import numpy as np
import tensorflow as tf

from midichar import encode_midi, decode_midi, encoded_notes_to_str, str_to_encoded_notes

from bpe import CharDataset


def main():
    data_dir = pathlib.Path('robbie-v1.0.0/clean/clean')
    filenames = glob.glob(str(data_dir/'*.mid*'))
    print('Number of files:', len(filenames))

    seq_length = 64

    with open('./output/training.txt', 'w') as inf:
        # string_from_file = inf.read()
        for i in range(len(filenames)):
            print(filenames[i])
            try:
                rn = encode_midi(filenames[i], 0, seq_length)
                mstr = encoded_notes_to_str(rn)
                inf.write(mstr)
            except Exception as e:
                print(e)
            if i >= 300:
                break


if __name__ == "__main__":
    main()

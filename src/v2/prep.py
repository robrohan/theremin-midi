import os
import pathlib
import glob

from midichar import encode_midi, encoded_notes_to_str


VERSION = os.environ["VERSION"]


def main():
    data_dir = pathlib.Path('./data/robbie-v1.0.0/clean')
    filenames = glob.glob(str(data_dir/'*.mid*'))
    print('Number of files:', len(filenames))

    # The size of the snippet of text to move from the midi data
    # to the training data
    seq_length = 256

    with open(f'./models/{VERSION}/training.txt', 'w') as inf:
        for i in range(len(filenames)):
            print(filenames[i])
            try:
                rn = encode_midi(filenames[i], 0, seq_length)
                mstr = encoded_notes_to_str(rn)
                inf.write(mstr + "\n")
            except Exception as e:
                print(e)
            # if i >= 900:
            #     break


if __name__ == "__main__":
    main()

import pathlib
import glob

from midichar import encode_midi, encoded_notes_to_str


def main():
    data_dir = pathlib.Path('robbie-v1.0.0/clean/clean')
    filenames = glob.glob(str(data_dir/'*.mid*'))
    print('Number of files:', len(filenames))

    seq_length = 128

    with open('./models/training.txt', 'w') as inf:
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

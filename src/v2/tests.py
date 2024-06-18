import logging
import numpy as np
from torch.utils.data import DataLoader
from midichar import (
    encode_midi,
    decode_midi,
    encoded_notes_to_str,
    str_to_encoded_notes,
)
from bpe import CharDataset, TextDataset


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def test_file():
    test_file = "./input/notes.mid"

    seq_length = 64

    logging.info(f"encoding midi file with a length of {seq_length}")
    raw_notes = encode_midi(test_file, 0, seq_length)

    logging.info("encoding the raw notes to utf8")
    midi_string = encoded_notes_to_str(raw_notes)

    logging.info("saving to a text file as plain text")
    with open("output/midi_as_text.txt", "w") as f:
        f.write(midi_string)

    #########################################################

    # string_from_file = ""
    # with open('output/midi_as_text.txt', 'r') as inf:
    #     string_from_file = inf.read()
    # block_size = 128
    # dataset = CharDataset(string_from_file, block_size)
    # print(dataset)

    #########################################################

    logging.info("reading plain text file back in")
    with open("output/midi_as_text.txt", "r") as inf:
        string_from_file = inf.read()

    logging.info("converting utf8 chars back into integers")
    from_file = str_to_encoded_notes(string_from_file)

    logging.info("creating new midi file from text file notes")
    decode_midi(
        from_file,
        out_file="./output/test2.midi",
        instrument_name=("Acoustic Grand Piano"),
    )

    # Example usage
    dataset = TextDataset(
        file_path="./models/training.txt",
        tokenizer_path="./models/miditok.model",
        max_length=64
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)


def main():
    test_file()


if __name__ == "__main__":
    configure_logger()
    main()

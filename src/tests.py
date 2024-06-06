from model import notes_to_midi, midi_to_notes
from model2 import encode_midi, decode_midi


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

    raw_notes = encode_midi(test_file, 0, 102400)
    print(raw_notes)
    decode_midi(
        raw_notes,
        out_file="./output/test2.midi",
        instrument_name=("Acoustic Grand Piano"),
    )


if __name__ == "__main__":
    main()

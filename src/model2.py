import pretty_midi
import pandas as pd


def encode_midi(midi_file: str, instrument_index=0,
                window_size=64) -> pd.DataFrame:
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[instrument_index]
        # notes = collections.defaultdict(list)
        notes = []

        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        i = 0
        for note in sorted_notes:
            start = note.start
            end = note.end

            step = start - prev_start
            velocity = note.velocity
            pitch = note.pitch
            duration = end - start

            int_step = round(255 * step)
            int_duration = round(255 * duration)

            # move the step of the note up to the high bits
            encoded_note = int_step << 24
            # Next high bits duration
            encoded_note += int_duration << 16
            # upper part of first 16 is velocity
            encoded_note += velocity << 8
            # lower part of first 16 is pitch
            encoded_note += pitch

            notes.append(encoded_note)
            
            prev_start = start
            i += 1
            if i >= window_size:
                break
    except Exception as e:
        print(f"could not load {midi_file} because {e}")
        return None

    return pd.DataFrame(notes)


def decode_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0.0
    for _, encoded_note in notes.iterrows():
        int_step = (encoded_note[0] >> 24) & 255
        int_duration = (encoded_note[0] >> 16) & 255
        velocity = (encoded_note[0] >> 8) & 255
        pitch = (encoded_note[0]) & 255

        # print(int_step / 255, int_duration / 255, velocity, pitch)
        f_step = (int_step / 255)*1000000/1000000
        f_duration = (int_duration / 255)*1000000/1000000

        start = float(prev_start + f_step)
        end = float(start + f_duration)

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

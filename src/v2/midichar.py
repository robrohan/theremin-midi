# import logging
import pretty_midi
import numpy as np
from typing import Any, Tuple


def encode_note(note: Any, prev_start: float) -> int:
    start = note.start
    end = note.end

    step = start - prev_start
    velocity = note.velocity
    pitch = note.pitch
    duration = (end - start)

    # This is whack, 0 wont work as it's not valid utf8 so really velocity is
    # only 1 to 3
    int_velocity = max(min(round(3*(round((velocity/127) * 100)/100)), 3), 1)  # 00
    int_step = min(round(63 * step), 63)                              # 00 0000
    int_duration = min(round(duration / 31 * 100), 31)                # 0000

    encoded_note = 0
    encoded_note += int_velocity << 24
    encoded_note += int_step << 16
    encoded_note += int_duration << 9
    # NOTE: Pitch will have to cross bytes into the upper part of the
    # 16 at some point, but for testing...
    encoded_note += pitch                       # XX00 0000

    # mark each section to pretend to be an actual utf8 character
    encoded_note = encoded_note | 4034955392
    if (encoded_note & 64) > 0:
        # if this bit is set we need to move it as it's needed for
        # UTF8 checking
        encoded_note = (encoded_note | (256))
    # then ensure it's flipped off to it checks out
    encoded_note = encoded_note & (~64)

    return encoded_note


def decode_note(encoded_note: int, prev_start: float) -> Tuple:
    # the top bit of the pitch got moved, move it back
    if (encoded_note & 256) > 0:
        encoded_note = (encoded_note & (~256))
        encoded_note = (encoded_note | 64)
        # print(f"{encoded_note:32b}")

    pitch = (encoded_note) & 127

    # remove utf8 encoding
    # encoded_note = encoded_note & (~4034955392)

    # print(f"{encoded_note:32b}")
    velocity = (encoded_note >> 24) & 3
    int_step = (encoded_note >> 16) & 63
    int_duration = (encoded_note >> 9) & 31

    f_step = (int_step / 63)*1000000/1000000
    f_duration = (int_duration / 31)*10
    velocity = int(127 * (velocity / 10.0))

    start = float(prev_start + f_step)
    end = float(start + f_duration)

    return (velocity, pitch, start, end,)


def encode_midi(midi_file: str, instrument_index=0,
                window_size=64) -> np.array:
    """
    Encode a midi file into a numpy array of integers using the
    instrument index with a max window size of window_size
    (window size is basically the number of note on events)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[instrument_index]
        notes = []
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        i = 0
        for note in sorted_notes:
            encoded_note = encode_note(note, prev_start)
            notes.append(encoded_note)
            prev_start = note.start
            i += 1
            if i >= window_size:
                break
    except Exception as e:
        print(f"could not load {midi_file} because {e}")
        return None

    return np.array(notes)  # pd.DataFrame(notes)


def decode_midi(
    notes: np.array,
    out_file: str,
    instrument_name: str = "Acoustic Grand Piano",
    bpm: int = 120
) -> pretty_midi.PrettyMIDI:
    """
    Given an array of encoded midi integers write a new midi file using
    the general midi instrument name as the instrument to use
    """

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    # Add a time signature change
    time_signature = pretty_midi.TimeSignature(numerator=4, denominator=4,
                                               time=0)
    pm.time_signature_changes.append(time_signature)

    # Key number according to [0, 11] Major, [12, 23] minor. For example, 0
    # is C Major, 12 is C minor - time is when to apply the change
    key_signature = pretty_midi.KeySignature(key_number=0, time=0)
    pm.key_signature_changes.append(key_signature)

    pm._tick_scales.append((0, 60/(bpm*pm.resolution)))
    pm._update_tick_to_time(0)

    prev_start = 0.0
    for _, encoded_note in enumerate(notes):
        velocity, pitch, start, end = decode_note(encoded_note, prev_start)
        if velocity == 0:
            velocity = 10
        # print(velocity, pitch, start, end)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start += (60000 / (bpm * 192)) / 4
        # prev_start = start

    pm.instruments.append(instrument)
    if out_file is not None:
        pm.write(out_file)
    return pm


def encoded_notes_to_str(raw_notes: np.array) -> str:
    midi_chars = []
    for _, raw_note in enumerate(raw_notes):
        try:
            midi_char = int(raw_note.astype(np.uint32))
            byte_array = midi_char.to_bytes(4, 'big')
            unicode_character = byte_array.decode('utf-8')
            midi_chars.append(unicode_character)
        except Exception as e:
            print(e)
            # print(f"{midi_char:32b} - {byte_array}")

    midi_string = "".join(midi_chars)
    return midi_string


def str_to_encoded_notes(string: str) -> np.array:
    from_file = []
    for c in string:
        from_file.append(int.from_bytes(bytes(c, "utf-8"), 'big'))
    return np.array(from_file)

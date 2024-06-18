"""
This file takes in a midi file removes the drums and collapses all the other
tracks onto one track.
"""
import sys
import hashlib
import logging
import mido
import glob
import pathlib
from pathlib import Path
import pretty_midi as pm


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def clean_with_pretty(out_path, file):
    fname = file.split('/')[-1]
    logging.debug(fname)
    hash = hashlib.sha1(bytes(fname, encoding='utf8'))

    my_file = Path(f'{out_path}/{hash.hexdigest()}.midi')
    if my_file.is_file():
        return

    midi = None
    try:
        midi = pm.PrettyMIDI(file)
    except Exception as e:
        logging.error(e)
        return

    inst_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    new_inst = pm.Instrument(program=inst_program)

    # check all instruments, remove them if not the instrument we want
    instruments_index = [
        i for i, inst in enumerate(midi.instruments) if inst.is_drum
    ]

    # remove all non drums, from the sorted such that no conflicting indexes
    for i in sorted(instruments_index, reverse=True):
        del midi.instruments[i]

    # combine all tracks into one
    for instrument in midi.instruments:
        #  if instrument.is_drum:
        for note in instrument.notes:
            new_inst.notes.append(note)

    # now delete all the tracks instruments
    instruments_index = [i for i, _ in enumerate(midi.instruments)]
    for i in sorted(instruments_index, reverse=True):
        del midi.instruments[i]

    # and add back our consolidated track
    midi.instruments.append(new_inst)

    midi.write(f'{out_path}/{hash.hexdigest()}.midi')


def main():
    dir_path = sys.argv[1]
    output_path = sys.argv[2]
    data_dir = pathlib.Path(dir_path)
    filenames = glob.glob(str(data_dir/'**/*.mid*'))

    logging.debug(data_dir)

    for i in filenames:
        file_path = i
        try:
            mid = mido.MidiFile(file_path)
        except Exception as e:
            print(e)
            continue

        # Get the MIDI version from the file header
        midi_version = mid.type
        print(f'MIDI version: {midi_version}')
        if midi_version == 0:
            pass
        elif midi_version == 1:
            logging.debug(file_path)
            clean_with_pretty(output_path, file_path)
        elif midi_version == 2:
            pass
        else:
            print('unhandled midi version')


if __name__ == "__main__":
    configure_logger()
    main()

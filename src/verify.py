import sys
import logging
import pretty_midi as pm
# from mido import Message, MidiFile, MidiTrack, MetaMessage


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def find_bad_files():
    try:
        mid1 = MidiFile(str(sys.argv[1]))
    except Exception as e:
        print(str(sys.argv[1]), "\t", e)


def main():
    # mid = MidiFile(str(sys.argv[1]))
    # logging.debug(sys.argv)

    midi = pm.PrettyMIDI(str(sys.argv[1]))

    inst_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    new_inst = pm.Instrument(program=inst_program)

    # check all instruments, remove them if not the instrument we want
    drum_instruments_index = [i for i, inst in enumerate(midi.instruments) if not inst.is_drum]

    # remove all non drums, from the sorted such that no conflicting indexes
    for i in sorted(drum_instruments_index, reverse=True):
        del midi.instruments[i]

    # combine all tracks into one
    for instrument in midi.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                new_inst.notes.append(note)

    print(new_inst)

    midi.write('test.mid')

    # mid_new = MidiFile(type=1, ticks_per_beat=mid.ticks_per_beat)
    # track_drums = MidiTrack()
    # mid_new.tracks.append(track_drums)

    # # Copy over all meta messages from the first track
    # for track in mid.tracks:
    #     for message in track:
    #         if isinstance(message, MetaMessage):
    #             if message.type not in ['lyrics', 'track_name', 'text', 'copyright']:
    #                 logging.debug(message)
    #                 track_drums.append(message)

    # for track in mid.tracks:
    #     for message in track:
    #         if message.type in ('note_on', 'note_off'):
    #             if message.channel == 9:
    #                 track_drums.append(message)
    #         elif message.type == 'program_change':
    #             if message.channel == 9 and message.program == 9:
    #                 track_drums.append(message)

    # mid_new.save("test.mid")

    # mid_new = MidiFile(type=1)
    # mid_new.ticks_per_beat = mid.ticks_per_beat

    # track_drums = MidiTrack(type=1)
    # # track_drums.name = "Percussion"
    # # Make sure we are doing drums
    # # track_drums.append(Message("program_change", program=9, time=0))

    # # Keep track of the notes that have been turned off
    # notes_off = set()

    # for track in mid.tracks:
    #     for message in track:
    #         if message.is_meta or message.type == "sysex":
    #             m = message.dict()
    #             if m["type"] in (
    #                 "sysex",
    #                 "time_signature",
    #                 "key_signature",
    #                 "set_tempo",
    #             ):
    #                 logging.debug(message)
    #                 track_drums.append(message)
    #             continue

    #         # if message.type == "note_on" or message.type == "note_off":
    #         #     m = message.dict()
    #             # logging.debug(message)

    #         try:
    #             m = message.dict()
    #             if m["channel"] == 9:
    #                 if message.type == "note_on":
    #                     # Only add note_on messages if the corresponding note_off message hasn't been added yet
    #                     if m["note"] not in notes_off:
    #                         logging.debug(message)
    #                         track_drums.append(message)
    #                 elif message.type == "note_off":
    #                     # Add note_off messages to the set of notes that have been turned off
    #                     notes_off.add(m["note"])
    #                     logging.debug(message)
    #                     track_drums.append(message)
    #         except Exception as e:
    #             logging.error(f"skipping {e} {message}")
    #             pass

    # # mid_new.tracks = mid.tracks[9]
    # # mid_new.tracks.append(track_drums)
    # mid_new.save("test.mid")
    # print("Input file ticks per beat:", mid.ticks_per_beat)
    # print("Output file ticks per beat:", mid_new.ticks_per_beat)


if __name__ == "__main__":
    configure_logger()
    main()
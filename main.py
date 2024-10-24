from argparse import ArgumentParser
import os
import glob
from pathlib import Path

import mir_eval
import pretty_midi as pm
import numpy as np

from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.features.quantisation import RNNJointQuantisationProcessor
from pm2s.features.hand_part import RNNHandPartProcessor
from pm2s.features.time_signature import CNNTimeSignatureProcessor
from pm2s.features.key_signature import RNNKeySignatureProcessor
from pm2s import CRNNJointPM2S


class Midi2Score:
    def __init__(self, dataset):
        self.dataset_path = dataset
        self.midi_files = []
        self._load_midi_files()
        self

    def _load_midi_files(self):
        """Load all MIDI files from the data directory."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Directory not found: {self.dataset_path}")
        print(self.dataset_path)
        self.midi_files = [
            Path(f) for f in glob.glob(f"{self.dataset_path}/**")
            if f.endswith('.mid') or f.endswith('.midi')
        ]
        
        if len(self.midi_files) == 0:
            raise ValueError("No MIDI files found in the provided directory.")
        
        print(f"Loaded {len(self.midi_files)} MIDI files from {self.dataset_path}")

    def process_dataset(self, method):
        function = getattr(self, method)
        for i, midi_data in enumerate(self.midi_files):
            print(midi_data)
            out = function(str(midi_data))
            print(out)

    def beat_tracking(self, midi_recording, generate_pianoroll=False):
        processor = RNNJointBeatProcessor()
        beats_pred, downbeats_pred = processor.process(midi_recording)
        if generate_pianoroll:
            self.visualize_beats(midi_recording, beats_pred, downbeats_pred)

    def visualize_beats(self, midi_recording, beats_pred, downbeats_pred):
        import matplotlib.pyplot as plt
        def get_piano_roll(midi_file, start_time, end_time):

            pr = np.zeros((128, int((end_time - start_time) * 100)))

            for instrument in pm.PrettyMIDI(midi_file).instruments:
                for note in instrument.notes:
                    if note.start >= end_time or note.end <= start_time:
                        continue
                    start = int((note.start - start_time) * 100)
                    end = int((note.end - start_time) * 100)

                    pr[note.pitch, start:end] = 1
            
            return pr

        start_time, end_time = 0, 120
        beats_pred_seg = beats_pred[np.logical_and(beats_pred >= start_time, beats_pred <= end_time)]
        downbeats_pred_seg = downbeats_pred[np.logical_and(downbeats_pred >= start_time, downbeats_pred <= end_time)]
        pr_seg = get_piano_roll(midi_recording, start_time, end_time)

        plt.figure(figsize=(20, 5))
        plt.imshow(1-pr_seg, aspect='auto', origin='lower', cmap='gray')
        for b in beats_pred_seg:
            plt.axvline(x=(b - start_time) * 100, ymin=0.75, ymax=1, color='green')
        for b in downbeats_pred_seg:
            plt.axvline(x=(b - start_time) * 100, ymin=0.5, ymax=1, color='blue')
        plt.xlabel('Time in milliseconds')
        plt.title('Upper (green): prediction, Lower (red): ground truth.')
        plt.show()
        plt.savefig('generated_score.png')

    def quantise_midi(self, midi_recording):
        processor = RNNJointQuantisationProcessor()
        onset_positions, note_values = processor.process(midi_recording)
        print('onset positions \t note values')
        print('-' * 50)
        for i in range(20):
            print('{:4f} \t\t {:.4f}'.format(onset_positions[i], note_values[i]))


    def predict_hand_part(self, midi_recording):
        processor = RNNHandPartProcessor()
        hand_parts = processor.process(midi_recording)
        print(hand_parts[:20])


    def predict_time_sig(self, midi_recording):
        processor_time_sig = CNNTimeSignatureProcessor()
        time_signature = processor_time_sig.process(midi_recording)
        # Single time signature prediction, assuming time signature does not change over the piece
        print("Time signature:")
        print(time_signature)

    def predict_key_sig(self, midi_recording):
        processor_key_sig = RNNKeySignatureProcessor()
        key_signature_changes = processor_key_sig.process(midi_recording)
        print("\nKey signature changes:")
        print(key_signature_changes)

    def performance_to_midi(self, perfm_midi_path, score_midi_path='generated_score.mid'):
        pm2s_processor = CRNNJointPM2S(
            beat_pps_args = {
                'prob_thresh': 0.5,
                'penalty': 1.0,
                'merge_downbeats': False,
                'method': 'dp',
            },
            ticks_per_beat = 480,
            notes_per_beat = [1, 6, 8],
        )
        # Convert and save the generated score midi
        pm2s_processor.convert(perfm_midi_path, score_midi_path, start_time=0, end_time=300)
        # To visualize the converted score MIDI, load it in a score annotation software such as MuseScore (you may need to add the corresponding time signature.)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--out_dir", type=str)
    # parser.add_argument("--")
    args = parser.parse_args()
    pm2s_converter = Midi2Score(args.in_dir)
    pm2s_converter.process_dataset(args.method)
    # python3 main.py --in_dir ../classical-music-research/data/raw/giantmidi/giantmidi_subset/ --method performance_to_midi
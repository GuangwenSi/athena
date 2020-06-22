# coding=utf-8
# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
import argparse
from collections import defaultdict
from pathlib import Path
import librosa
import numpy as np
from python_speech_features import fbank
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

class Audio:
    """if audio_dir is not None, then all the 'ext' file will get mfcc and be saved as npy
       all ext file in os.path.join(cache_dir, 'audio-fbanks') will be made as a dict to get
       all dir of a speaker
    """
    def __init__(self, working_dir, audio_dir=None, sample_rate=16000, num_fbanks=64, train_test_ratio=0.8, ext='flac'):
        self.ext = ext
        self.train_test_ratio = train_test_ratio
        self.working_dir = working_dir
        self.cache_dir = os.path.join(working_dir, 'npy-fbanks')
        self.ensures_dir(self.cache_dir)
        if audio_dir is not None: # get mfcc and save as npy
            self.build_cache(os.path.expanduser(audio_dir), sample_rate, num_fbanks)

        self.all_npy_files = self.find_files(self.cache_dir, ext='npy')
        self.split_ids = [Path(cache_file).stem.split('_') for cache_file in self.all_npy_files]

        self.speakers_to_utterances = defaultdict(dict)
        for i, split_id in enumerate(self.split_ids):
            self.speakers_to_utterances[split_id[0]][split_id[1]] = self.all_npy_files[i]

        self.speaker_ids = sorted(self.speakers_to_utterances)

    def build_cache(self, audio_dir, sample_rate, num_fbanks):
        audio_files = self.find_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any {self.ext} files in {audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in {audio_dir}.')
        with tqdm(audio_files) as bar:
            for audio_filename in bar:
                bar.set_description(audio_filename)
                self.cache_audio_file(audio_filename, sample_rate, num_fbanks)

    def cache_audio_file(self, input_filename, sample_rate, num_fbanks):
        sp, utt = self.extract_speaker_and_utterance_ids(input_filename)
        cache_filename = os.path.join(self.cache_dir, f'{sp}_{utt}.npy')
        if not os.path.isfile(cache_filename):
            try:
                mfcc = self.read_mfcc(input_filename, sample_rate, num_fbanks)
                np.save(cache_filename, mfcc)
            except librosa.util.exceptions.ParameterError as e:
                logger.error(e)

    @staticmethod
    def trim_silence(audio, threshold):
        """Removes silence at the beginning and end of a sample."""
        energy = librosa.feature.rms(audio)
        frames = np.nonzero(np.array(energy > threshold))
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        audio_trim = audio[0:0]
        left_blank = audio[0:0]
        right_blank = audio[0:0]
        if indices.size:
            audio_trim = audio[indices[0]:indices[-1]]
            left_blank = audio[:indices[0]]  # slice before.
            right_blank = audio[indices[-1]:]  # slice after.
        return audio_trim, left_blank, right_blank

    @staticmethod
    def read(filename, sample_rate):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

    def extract_speaker_and_utterance_ids(self, filename: str):
        # 'audio/dev-other/116/288045/116-288045-0000.flac'
        speaker, _, basename = Path(filename).parts[-3:]
        filename.split('-')
        utterance = os.path.splitext(basename.split('-', 1)[-1])[0]
        assert basename.split('-')[0] == speaker
        return speaker, utterance

    def read_mfcc(self, input_filename, sample_rate, num_fbanks):
        audio = self.read(input_filename, sample_rate)
        energy = np.abs(audio)
        silence_threshold = np.percentile(energy, 95)
        offsets = np.where(energy > silence_threshold)[0]
        audio_voice_only = audio[offsets[0]:offsets[-1]]
        mfcc = self.mfcc_fbank(audio_voice_only, sample_rate, num_fbanks)
        return mfcc

    def mfcc_fbank(self, signal: np.array, sample_rate: int, num_fbanks):
        filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=num_fbanks)
        frames_features = self.normalize_frames(filter_banks)
        return np.array(frames_features, dtype=np.float32)

    def normalize_frames(self, m, epsilon=1e-12):
        return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]

    def train_test_sp_to_utt(self, is_test):
        sp_to_utt = {}
        for speaker_id, utterances in self.speakers_to_utterances.items():
            utterances_files = sorted(utterances.values())
            train_test_sep = int(len(utterances_files) * self.train_test_ratio)
            sp_to_utt[speaker_id] = utterances_files[train_test_sep:] if is_test else utterances_files[:train_test_sep]
        return sp_to_utt

    def creat_csv(self, is_test, total_csv):
        csv_name = 'test.csv' if is_test else 'train.csv'
        with open(os.path.join(self.working_dir, csv_name), 'w') as f:
            f.write("original_id\tmfcc_dir\n")
            total_f = open(total_csv)
            sp_to_utt = self.train_test_sp_to_utt(is_test=is_test)
            desc = f'Converting to csv [{"test" if is_test else "train"}]'
            for i, speaker_id in enumerate(tqdm(self.speaker_ids, desc=desc)):
                utterances_files = sp_to_utt[speaker_id]
                for j in utterances_files:
                    _data = speaker_id + '\t' + j + '\n'
                    f.write(_data)
                    total_f.write(_data)
            total_f.close()

    def find_files(self, audio_dir, ext='npy'):
        return sorted(glob(audio_dir + f'/**/*.{ext}', recursive=True))

    def ensures_dir(self, directory):
        if len(directory) > 0 and not os.path.exists(directory):
            os.makedirs(directory)


def main(args):
    working_dir = args.working_dir
    audio_dir = args.audio_dir
    sample_rate = args.sample_rate
    num_fbanks = args.num_fbanks
    train_test_ratio = args.train_test_ratio
    # csv for save mfcc
    total_train_csv = os.path.join(working_dir,'train.csv')
    total_test_csv = os.path.join(working_dir, 'test.csv')
    with open(total_train_csv, 'w') as train_file:
        train_file.write("original_id\tmfcc_dir\n")
    with open(total_test_csv, 'w') as test_file:
        test_file.write("original_id\tmfcc_dir\n")
    #extract mfcc and creat csv for every sub set
    sub_dirs = ['dev-clean', 'dev-other', 'test-clean', 'test-other', \
                                'train-clean-100', 'train-clean-360', 'train-other-500']
    for sub_dir in sub_dirs:
        # extract fbank, save as npy, and make csv for every sub dir
        sub_working_dir = os.path.join(working_dir, sub_dir)
        sub_audio_dir = os.path.join(audio_dir, sub_dir)
        sub_audio = Audio(working_dir=sub_working_dir, audio_dir=sub_audio_dir, sample_rate=sample_rate, \
              num_fbanks=num_fbanks, train_test_ratio=train_test_ratio, ext='flac')
        # original_id_to_sorted = dict(zip(sorted(sub_audio.speakers_to_utterances)))# a dict to map id
        sub_audio.creat_csv(is_test=False, total_csv=total_train_csv)
        sub_audio.creat_csv(is_test=True, total_csv=total_test_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', default='')
    parser.add_argument('--audio_dir', default='')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--num_fbanks', type=int, default=64)
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    args = parser.parse_args()
    main(args)
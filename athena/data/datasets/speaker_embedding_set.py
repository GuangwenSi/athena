# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao
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
# pylint: disable=no-member, invalid-name
""" audio dataset """

import random
import math
from absl import logging
import tensorflow as tf
from ...utils.hparam import register_and_parse_hparams
from .base import BaseDatasetBuilder
from athena.transform import AudioFeaturizer
import numpy as np
from random import choice


class SpeakerEmbeddingSoftmaxDatasetBuilder(BaseDatasetBuilder):
    """ SpeakerEmbeddingDatasetBuilder

    Args:
        for __init__(self, config=None)

    Config:
        feature_config: the config file for feature extractor, default={'type':'Fbank'}
        data_csv: the path for original LDC HKUST,
            default='/tmp-data/dataset/opensource/hkust/train.csv'
        force_process: force process, if force_process=True, we will re-process the dataset,
            if False, we will process only if the out_path is empty. default=False

    Interfaces::
        __len__(self): return the number of data samples

        @property:
        sample_shape:
            {"input": tf.TensorShape([None, self.audio_featurizer.dim,
                                  self.audio_featurizer.num_channels]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, self.audio_featurizer.dim *
                                  self.audio_featurizer.num_channels]),}
    """

    default_config = {
        "data_csv": None,
        "acoustic_dim": 64,
        "max_frame": 160,
        "counts_per_speaker": 600,
        "num_for_epoch": 640000
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        if self.hparams.data_csv is not None:
            self.load_csv(self.hparams.data_csv)

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_path):
        """Generate a list of tuples (wav_filename, wav_length_ms, speaker)."""
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        lines = lines[1:]

        self.speakers_to_utterances = {}
        for line in lines:
            _real_id, utterances = line.split("\t")
            if _real_id not in self.speakers_to_utterances:
                self.speakers_to_utterances[_real_id] = []
            self.speakers_to_utterances[_real_id].append(utterances)
        self.speakers = sorted(self.speakers_to_utterances.keys())
        self.id_map = dict(zip(self.speakers, range(len(self.speakers))))

        self.entries = []
        for k in self.speakers_to_utterances:
            utterances_files = self.speakers_to_utterances[k]
            for j, utterance_file in enumerate(np.random.choice(utterances_files,
                                                size=self.hparams.counts_per_speaker, replace=True)):
                self.entries.append(tuple([k, self.id_map[k], utterance_file]))
        #self.entries.sort(key=lambda item: int(item[1]))
        return self

    def load_csv(self, file_path):
        """ load csv file """
        return self.preprocess_data(file_path)

    def __getitem__(self, index):
        original_speaker_id, sorted_speaker_id, acoustic_feature_dir = self.entries[index]
        acoustic_feature = np.load(acoustic_feature_dir)
        mfcc = self.sample_from_mfcc(acoustic_feature, self.hparams.max_frame)

        return {
            "input": mfcc,
            "output": np.array([int(sorted_speaker_id)])
        }

    def sample_from_mfcc(self, mfcc, max_length):
        if mfcc.shape[0] >= max_length:
            r = choice(range(0, len(mfcc) - max_length + 1))
            s = mfcc[r:r + max_length]
        else:
            s = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
        return np.expand_dims(s, axis=-1)

    def __len__(self):
        """ return the number of data samples """
        return len(self.entries)

    @property
    def num_class(self):
        """ return the max_index of the vocabulary """
        target_dim = len(self.speakers)
        return target_dim

    @property
    def sample_type(self):
        return {
            "input": tf.float32,
            "output": tf.int32,
        }

    @property
    def sample_shape(self):
        return {
            "input": tf.TensorShape([None, self.hparams.acoustic_dim, 1]),
            "output": tf.TensorShape([1]),
        }

    @property
    def sample_signature(self):
        return (
            {
                "input": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            },
        )

    def compute_cmvn_if_necessary(self, is_necessary=True):
        pass


class SpeakerEmbeddingTripletDatasetBuilder(SpeakerEmbeddingSoftmaxDatasetBuilder):
    """for triplet loss speaker embedding training"""
    def preprocess_data(self, file_path):
        """Generate a list of tuples (wav_filename, wav_length_ms, speaker)."""
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        lines = lines[1:]

        self.speakers_to_utterances = {}
        for line in lines:
            _real_id, utterances = line.split("\t")
            if _real_id not in self.speakers_to_utterances:
                self.speakers_to_utterances[_real_id] = []
            self.speakers_to_utterances[_real_id].append(utterances)
        self.speakers = sorted(self.speakers_to_utterances.keys())
        self.id_map = dict(zip(self.speakers, range(len(self.speakers))))
        self.id_map_inverse = dict(zip(range(len(self.speakers)), self.speakers))
        self.creat_a_epoch_data1()
        return self

    def creat_a_epoch_data1(self):
        entries = []
        random.shuffle(self.speakers)
        #selected_speakers = self.speakers[:640]
        for i in range(self.hparams.num_for_epoch):
            anchor_speaker = np.random.choice(self.speakers, size=1, replace=False)[0]
            negative_speaker = np.random.choice(list(set(self.speakers) - {anchor_speaker}), size=1)[0]
            anchor_dir, pos_dir = np.random.choice(self.speakers_to_utterances[anchor_speaker], size=2, replace=False)
            neg_dir = np.random.choice(self.speakers_to_utterances[negative_speaker], size=1, replace=True)[0]
            i_data = tuple([self.id_map[anchor_speaker], anchor_dir, pos_dir, neg_dir])
            entries.append(i_data)
        self.entries = entries

    def creat_a_epoch_data2(self, m=None):
        if m is not None:
            self.m = m
        entries = []
        random.shuffle(self.speakers)
        num_speaker = len(self.speakers) if len(self.speakers) < 640 else 640
        num_per_speaker = self.hparams.num_for_epoch // num_speaker
        anchor_speakers = np.random.choice(self.speakers, size=num_speaker, replace=False)

        embedding_dict = {}
        utterance_dict = {}
        for speaker in anchor_speakers:
            utterance_dirs = np.random.choice(self.speakers_to_utterances[speaker], size=num_per_speaker, replace=True)
            tmp_mfcc = np.array(
                [self.sample_from_mfcc(np.load(utterance), self.hparams.max_frame) for utterance in utterance_dirs])
            tmp_embedding = self.m.predict(tmp_mfcc)
            embedding_dict[speaker] = tmp_embedding
            utterance_dict[speaker] = utterance_dirs
        for k in embedding_dict.keys():
            for anc in range(len(embedding_dict[k])):
                anc_embedding = embedding_dict[k][anc]
                anc_utterance = utterance_dict[k][anc]

                neg_speaker = np.random.choice(list(set(anchor_speakers) - {k}), size=1)[0]
                neg_embedding = embedding_dict[neg_speaker]
                anchor_cos = batch_cosine_similarity([anc_embedding]*len(neg_embedding), neg_embedding)
                select_neg = np.argsort(anchor_cos)[-1]
                neg_utterance = utterance_dict[neg_speaker][select_neg]

                pos_indexes = [j for (j, a) in enumerate(utterance_dict[k]) if a != anc_utterance]
                pos_embedding = embedding_dict[k][pos_indexes]
                pos_cos = batch_cosine_similarity([anc_embedding] * len(pos_embedding), pos_embedding)
                select_pos = pos_indexes[np.argsort(pos_cos)[0]]
                pos_utterance = utterance_dict[k][select_pos]

                entries.append([self.id_map[k], anc_utterance, pos_utterance, neg_utterance])
        self.entries = entries

    def __getitem__(self, index):
        anchor_speaker, anchor_dir, pos_dir, neg_dir = self.entries[index]
        anchor_mfcc = np.load(anchor_dir)
        pos_mfcc = np.load(pos_dir)
        neg_mfcc = np.load(neg_dir)

        anchor_mfcc = self.sample_from_mfcc(anchor_mfcc, self.hparams.max_frame)
        pos_mfcc = self.sample_from_mfcc(pos_mfcc, self.hparams.max_frame)
        neg_mfcc = self.sample_from_mfcc(neg_mfcc, self.hparams.max_frame)

        mfcc = np.concatenate([anchor_mfcc, pos_mfcc, neg_mfcc], axis=0)
        return {
            "input": mfcc,
            "output": np.array([int(anchor_speaker)])
        }

    def batch_wise_shuffle(self, batch_size=64):
        if len(self.entries) == 0:
            return self
        self.creat_a_epoch_data2()
        return self

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)
    # l1 = np.sum(np.multiply(x1, x1),axis=1)
    # l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s
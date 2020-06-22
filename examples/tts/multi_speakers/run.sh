# coding=utf-8
# Copyright (C) ATHENA AUTHORS
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

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

source tools/env.sh

stage=1
stop_stage=100
horovod_cmd="horovodrun -np 4 -H localhost:4"
horovod_prefix="horovod_"

dataset_dir="/nfs/cold_project/datasets/opensource_data/librispeech"
sample_rate=16000
num_fbank=64
#num_frames=160
counts_per_speaker_train=600
counts_per_speaker_test=100
train_test_ratio=0.8

mkdir -p $dataset_dir


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    echo "download data ..."
    . ./examples/tts/multi_speakers/local/download_data.sh $dataset_dir || exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # prepare data and make csv for json config
    echo "prepare data..."
    python examples/tts/multi_speakers/local/prepare_data.py \
        --working_dir "${dataset_dir}/audio-fbanks" \
        --audio_dir "${dataset_dir}/LibriSpeech" \
        --sample_rate "$sample_rate" \
        --num_fbanks "$num_fbank" \
        --train_test_ratio "$train_test_ratio" || exit 1

    cp "${dataset_dir}/audio-fbanks/train.csv" "examples/tts/multi_speakers/data/ds_train.csv"
    cp "${dataset_dir}/audio-fbanks/test.csv" "examples/tts/multi_speakers/data/ds_est.csv"

    cp "${dataset_dir}/audio-fbanks/train-clean-360/train.csv" "examples/tts/multi_speakers/data/ds_train_pre.csv"
    cp "${dataset_dir}/audio-fbanks/train-clean-360/test.csv" "examples/tts/multi_speakers/data/ds_test_pre.csv"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # pre-train
    echo "pre-train..."
    $horovod_cmd python athena/${horovod_prefix}main.py
        examples/tts/multi_speakers/configs/tts_multi_speakers_pretrain.json || exit 1
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # triplet-train
    echo "triplet-train..."
    $horovod_cmd python athena/${horovod_prefix}main.py
        examples/tts/multi_speakers/configs/tts_multi_speakers.json || exit 1
fi
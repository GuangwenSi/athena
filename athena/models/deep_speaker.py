# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

import tensorflow as tf
from absl import logging
from ..utils.hparam import register_and_parse_hparams
from .base import BaseModel

layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses

K = tf.keras.backend

class DeepSpeakerSoftmax(BaseModel):
    """ a sample implementation of deep speaker """
    default_config = {
        "hidden_size": 512,
        "num_layers": [64, 128, 256, 512],
        "rate": 0.1,
        "include_softmax":False,
        "alpha": 0.1
    }
    def __init__(self, data_descriptions, config=None, train_mode=False):
        super().__init__()
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.hidden_size = self.hparams.hidden_size
        self.include_softmax = self.hparams.include_softmax
        self.num_speakers_softmax = data_descriptions.num_class
        self.data_descriptions = data_descriptions
        #self.step_of_model = 1
        self.clipped_relu_count = 0

        input_feature = layers.Input(
            shape=data_descriptions.sample_shape["input"],
            dtype=tf.float32
        )# shape is [None, 160, 64, 1]   None is: total_num_speaker * wav_num_per_speaker , for examples (5*600, 160, 64, 1)
        x = self.cnn_component(input_feature) # shape is (3000, 10, 4, 512) # the last 2 dimension must be 4 and 512
        x = layers.Reshape((-1, 2048))(x) # shape is (3000, 10, 2048)
        x = layers.Lambda(lambda y: K.mean(y, axis=1), name='average')(x) # frame-level to uttrance-level
        if self.include_softmax:
            x = layers.Dropout(self.hparams.rate)(x)# used for softmax because the dataset we pre-train on might be too small. easy to overfit.
        x = layers.Dense(self.hidden_size, name='affine')(x)
        if self.include_softmax:
            x = layers.Dense(self.num_speakers_softmax, activation='softmax')(x)
        else:
            x = layers.Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
        self.m = tf.keras.Model(inputs=input_feature, outputs=x) # tf.keras.models.Model
        logging.info(self.m.summary())

    def cnn_component(self, inp):
        for i in range(len(self.hparams.num_layers)):
            inp = self.conv_and_res_block(inp, self.hparams.num_layers[i], stage=i+1) # in: (3000, 160, 64, 1) out:(3000, 80, 32, 64)
        return inp

    def conv_and_res_block(self, inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = layers.Conv2D(filters,
                          kernel_size=5,
                          strides=2,
                          activation=None,
                          padding='same',
                          kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name)(inp)# in: (3000, 160, 64, 1) out: (3000, 80, 32, 64)
        o = layers.BatchNormalization(name=conv_name + '_bn')(o) # (3000, 80, 32, 64)
        o = self.clipped_relu(o) # (3000, 80, 32, 64)
        for i in range(3):
            o = self.identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o # (3000, 80, 32, 64)

    def clipped_relu(self, inputs):
        _relu = layers.Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name=f'clipped_relu_{self.clipped_relu_count}')(inputs)
        self.clipped_relu_count += 1
        return _relu

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name_base = f'res{stage}_{block}_branch'

        x = layers.Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2a')(input_tensor) # in [3000, 80, 32, 64] out [3000, 80, 32, 64]
        x = layers.BatchNormalization(name=conv_name_base + '_2a_bn')(x)
        x = self.clipped_relu(x)

        x = layers.Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2b')(x)
        x = layers.BatchNormalization(name=conv_name_base + '_2b_bn')(x)
        x = self.clipped_relu(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def call(self, samples, training=None):
        """ call function """
        #self.step_of_model += 1
        return self.m(samples["input"], training=training)

    def get_loss(self, logits, samples, training=None):
        # sparse_categorical_crossentropy loss with adam optimizer
        """
        res = logits - samples['output']
        loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(res * res, axis=2), axis=1))
        """

        loss = losses.sparse_categorical_crossentropy(y_true=samples['output'],
                                                   y_pred=logits,
                                                   axis=-1)
        loss = tf.reduce_mean(loss, axis=0)
        self.metric.update_state(loss)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics


class DeepSpeakerTriplet(DeepSpeakerSoftmax):
    """ use triplet loss instead of softmax """
    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        if model_type == "":
            return
        if model_type == "mpc":
            logging.info("loading from pretrained mpc model")
            self.x_net = pretrained_model.x_net
            self.transformer.encoder = pretrained_model.encoder
        elif model_type == "SpeechTransformer":
            logging.info("loading from pretrained SpeechTransformer model")
            self.x_net = pretrained_model.x_net
            self.y_net = pretrained_model.y_net
            self.transformer = pretrained_model.transformer
            self.final_layer = pretrained_model.final_layer
        elif model_type == "deep_speaker_softmax":
            logging.info("loading from pretrained DeepSpeaker model")
            for layer_i in range(len(self.m.layers[:-2])):
                self.m.layers[layer_i].set_weights(pretrained_model.m.layers[layer_i].get_weights())
            self.m.layers[-2].set_weights(pretrained_model.m.layers[-2].get_weights())
        else:
            raise ValueError("NOT SUPPORTED")

        self.data_descriptions.creat_a_epoch_data2(self.m)  # select hard triplets data


    def call(self, samples, training=None):
        """ call function """
        #self.step_of_model += 1
        #if self.step_of_model % 100 == 0:
        #    self.data_descriptions.creat_a_epoch_data2(self.m)
        return self.m(samples["input"], training=training)

    def prepare_samples(self, samples):
        """ for special data prepare
        carefully: do not change the shape of samples
        """
        triplet_input = samples['input']
        _batch = tf.shape(triplet_input)[0]
        _dim = tf.shape(triplet_input)[2]

        triplet_input = tf.reshape(triplet_input,[_batch,3,-1,_dim,1])
        triplet_input = tf.concat(
            [triplet_input[:,0,:,:,:],triplet_input[:,1,:,:,:],triplet_input[:,2,:,:,:]],
            axis=0
        )

        samples['input'] = triplet_input
        return samples

    def get_loss(self, logits, samples, training=None):
        split = K.shape(logits)[0] // 3
        anchor = logits[0:split]
        positive_ex = logits[split:2 * split]
        negative_ex = logits[2 * split:]

        sap = self.batch_cosine_similarity(anchor, positive_ex)
        san = self.batch_cosine_similarity(anchor, negative_ex)
        loss = K.maximum(san - sap + self.hparams.alpha, 0.0)
        loss = K.mean(loss)

        self.metric.update_state(loss)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics

    def batch_cosine_similarity(self, x1, x2):
        dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
        return dot
import numpy as np

# hide tensorflow infos, warnings, or errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore',category=UserWarning)

import tensorflow as tf

# hide more tensorflow warnings
tf.get_logger().setLevel('ERROR')
# Disable eager execution for performance
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Input, Flatten, Dense, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class ActionValueEstimator:
    def __init__(self, env, learning_rate = 3e-4):
        self._observation_shape = env.observation_space.shape
        self._nr_actions = env.action_space.n
        
        self._model_multi, self._model_single = self._build_models(learning_rate)
        
    def _build_models(self, learning_rate):
        observation = Input(self._observation_shape, name='observation')

        h = Flatten(name='flatten')(observation)
        h = Dense(64, activation='relu', name='dense1')(h)
        h = Dense(64, activation='relu', name='dense2')(h)
        h = Dense(32, activation='relu', name='dense3')(h)
        action_values = Dense(self._nr_actions, activation='linear', name='action_values')(h)
        model_multi = Model(observation, action_values, name='action_values')

        action_one_hot = Input(self._nr_actions, name='action_one_hot')
        selected_action_value = Multiply(name='select')([action_values, action_one_hot])
        action_value = Lambda(lambda x: tf.keras.backend.sum(x, axis=1, keepdims=True), name='action_value')(selected_action_value)
                
        model_single = Model([observation, action_one_hot], action_value, name='action_value')
        model_single.compile(loss='mse', optimizer=Adam(learning_rate))
        
        return model_multi, model_single

    def predict_on_batch(self, observations):
        return self._model_multi.predict(observations)

    def predict(self, observation):
        return self.predict_on_batch(np.expand_dims(observation, axis=0))[0]

    def train_on_batch(self, observations, actions, values):
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self._nr_actions)
        self._model_single.train_on_batch([observations, actions_one_hot], values)

    def load_weights(self, weights_file):
        self._model_single.load_weights(weights_file)

    def save_weights(self, weights_file):
        self._model_single.save_weights(weights_file)


class StateValueEstimator:
    def __init__(self, env, learning_rate = 1e-3):
        self._observation_shape = env.observation_space.shape
        
        self._model = self._build_models(learning_rate)
        
    def _build_models(self, learning_rate):
        observation = Input(self._observation_shape, name='observation')

        h = Flatten(name='flatten')(observation)
        h = Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense1')(h)
        h = Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense2')(h)
        h = Dense(32, activation='relu', kernel_initializer='he_uniform', name='dense3')(h)
        state_value = Dense(1, activation='linear', kernel_initializer='uniform', name='state_value')(h)

        model = Model(observation, state_value, name='state_value')
        model.compile(loss='mse', optimizer=Adam(learning_rate))
        return model

    def predict_on_batch(self, observation):
        return self._model.predict(observation)

    def predict(self, observation):
        return self.predict_on_batch(np.expand_dims(observation, axis=0))[0]

    def train_on_batch(self, observations, values):
        self._model.train_on_batch(observations, values)

    def load_weights(self, weights_file):
        self._model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self._model.save_weights(weights_file)

class PolicyEstimator:
    def __init__(self, env, learning_rate = 3e-4):
        self._observation_shape = env.observation_space.shape
        self._nr_actions = env.action_space.n
        
        self._model = self._build_model(learning_rate)
        
    def _build_model(self, learning_rate):
        observation = Input(self._observation_shape, name='observation')

        h = Flatten(name='flatten')(observation)
        h = Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense1')(h)
        h = Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense2')(h)
        h = Dense(32, activation='relu', kernel_initializer='he_uniform', name='dense3')(h)
        action_probs = Dense(self._nr_actions, activation='softmax', kernel_initializer='glorot_normal', name='action_probs')(h)

        model = Model(observation, action_probs, name='action_probs')
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate))
        return model

    def predict_on_batch(self, observation):
        return self._model.predict(observation)

    def predict(self, observation):
        return self.predict_on_batch(np.expand_dims(observation, axis=0))[0]

    def train_on_batch(self, observation, actions, advantages):
        advantages = (advantages - advantages.mean()) / advantages.std() # normalize
        self._model.train_on_batch(observation, actions, sample_weight=advantages.squeeze(-1))

    def load_weights(self, weights_file):
        self._model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self._model.save_weights(weights_file)

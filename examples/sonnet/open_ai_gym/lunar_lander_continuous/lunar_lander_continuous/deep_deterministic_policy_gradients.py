import numpy as np
from copy import deepcopy

import tensorflow as tf
import sonnet as snt

class MLP(snt.Module):

    def __init__(self, hidden1_dim, hidden2_dim, final_dim, activation, final_activation):
        super(MLP, self).__init__()
        self.hidden1 = snt.Linear(hidden1_dim, name="hidden1")
        self.activation1 = tf.keras.layers.Activation(activation)
        self.hidden2 = snt.Linear(hidden2_dim, name="hidden2")
        self.activation2 = tf.keras.layers.Activation(activation)
        self.final = snt.Linear(final_dim, name="final")
        self.final_activation = tf.keras.layers.Activation(final_activation)

    def __call__(self, inpt):
        tmp = self.hidden1(inpt)
        tmp = self.activation1(tmp)
        tmp = self.hidden2(tmp)
        tmp = self.activation2(tmp)
        tmp = self.final(tmp)
        outpt = self.final_activation(tmp)
        return outpt


class DeterministicMLPActor(snt.Module):

    def __init__(self, act_dim, act_limit, hidden1_dim, hidden2_dim, activation):
        super(DeterministicMLPActor, self).__init__()
        self.pi = MLP(hidden1_dim, hidden2_dim, final_dim=act_dim, activation=activation, final_activation='tanh')
        self.act_limit = act_limit

    def __call__(self, obs):
        return self.act_limit * self.pi(obs)

class MLPQFunction(snt.Module):

    def __init__(self, hidden1_dim, hidden2_dim, activation):
        super(MLPQFunction, self).__init__()
        self.q = MLP(hidden1_dim, hidden2_dim, final_dim=1, activation=activation, final_activation='linear')
    
    def __call__(self, obs, act):
        inpt = tf.concat([obs, act], axis=1)
        q = self.q(inpt)
        return tf.squeeze(q, axis=1)



q = MLPQFunction(256, 128, activation='relu')
obs = np.random.rand(10, 8)
act = np.random.rand(10, 4)
obs = tf.convert_to_tensor(obs)
act = tf.convert_to_tensor(act)
print(q(obs, act))

print(snt.format_variables(q.variables))

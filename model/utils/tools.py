# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from progressbar import Bar, Percentage, ProgressBar, Timer
from sklearn.preprocessing import OneHotEncoder


def load_csi(directory, gests, runs):
    """
    Loads all the CSI data specified by the parameters.

    Parameters
    ----------
    directory: str
        Head directory containing all the CSI data
        "/home/radomir/har/esp32/data/"
    gests: list
        Which gestures/classes to include in data
        ["lr", "pp", "ng"]
    runs: list
        Which runs to include in data
        [0, 1, 2, 3]
    
    Returns
    -------
    x: list
        x values of dataset
    y: list
        y values of dataset
    """
    
    x = []
    y = []
    max = len(runs)*60
    # pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=max).start()

    for i, run in enumerate(runs):
        for j, gest in enumerate(gests):
            for k in range(20):
                # load .mat file with CSI
                csi = loadmat(f"{directory}{gest}_run{run}_rep{k}")["window"]
                
                # do any processing necessary to the CSI data here
                # extract active subcarriers
                active_sc = list(range(6,32)) + list(range(33,59))
                csi = csi[:, active_sc]

                # normalize
                csi = (csi - np.mean(csi))/ np.std(csi)
                # csi = (csi - np.mean(csi, axis=0)[np.newaxis, :]) / np.std(csi, axis=0)[np.newaxis, :]

                # append to data
                x.append(np.array(csi))
                # append index as label
                y.append(j)
                
                # pbar.update(1 + k + j*20 + i*len(gests)*20)
    
    # pbar.finish()

    x = np.array(x)
    y = np.array(y).reshape(-1,1)
    # encode classes
    label_encoder = OneHotEncoder(sparse=False)
    y = label_encoder.fit_transform(y)

    return x, y


def marginLoss(y_true, y_pred):
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1
    
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
    lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def plot_model(history, dir):
    plt.plot(history.history["capshar_accuracy"])
    plt.plot(history.history["val_capshar_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.grid()
    # plt.show()
    plt.savefig(f"{dir}mdl_accuracy.png", bbox_inches="tight")
    plt.close()
    
    plt.plot(history.history["capshar_loss"])
    plt.plot(history.history["val_capshar_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.grid()
    # plt.show()
    plt.savefig(f"{dir}mdl_loss.png", bbox_inches="tight")
    plt.close()


def load_model_predictions_generator(x_test, batch_size, model):
    t_start = time.time()
    predictions, x_recon = model.predict(x_test, verbose=0, batch_size=batch_size)
    inference_time = time.time() - t_start

    return predictions, inference_time


def create_conf_matrix(y_test, predictions):
    matrix = confusion_matrix(np.argmax(y_test,1), np.argmax(predictions,1), normalize="true").astype(float)
    matrix = np.round(matrix,3) * 100

    return matrix


def plot_conf_matrix(matrix, file, title=""):
    fig=plt.figure(figsize=(6,5))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, fmt="g", ax=ax)
    acc = np.round(np.mean(matrix.diagonal()), 1)

    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(f"Confusion Matrix - Accuracy: {acc}%" + title)
    ax.xaxis.set_ticklabels(["lr", "pp", "ng"])
    ax.yaxis.set_ticklabels(["lr", "pp", "ng"])

    plt.savefig(file, bbox_inches="tight")
    plt.close()
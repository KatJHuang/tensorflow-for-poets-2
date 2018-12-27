from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa
import numpy as np
import tensorflow as tf
from scipy.signal import butter, lfilter

"""
===========================================================================
Load graph, obtain tensors needed for prediction 
===========================================================================
"""

# keep all weights when evaluating the graph
KEEP_ALL = 1

# names of tensor to feed input to and fetch result from
IMG_BUFFER = 'DecodeJpeg:0'
FINAL_RESULT = 'final_result:0'
KEEP_PROB = 'final_training_ops/dropout/keep_prob:0'


def load_graph(file_name):
    """
    Load the trained graph
    :param file_name:
    :return:
    """
    with open(file_name,'rb') as f:
        content = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def get_tensors_for_prediction(filename):
    """
    Bundle the graph and tensors needed to run prediction
    :param filename:
    :return:
    """
    with load_graph(filename).as_default() as graph:

        image_buffer_input = graph.get_tensor_by_name(IMG_BUFFER)
        final_tensor = graph.get_tensor_by_name(FINAL_RESULT)
        keep_prob = graph.get_tensor_by_name(KEEP_PROB)

        with tf.name_scope('prediction'):
            prediction = tf.argmax(final_tensor, 1)

    return graph, image_buffer_input, prediction, keep_prob


def predict(graph, image_buffer_input, prediction, keep_prob, img):
    """
    Predicts the class for the InceptionV3 model
    :param img: image of 3 channels
    :param graph: trained graph loaded from drive
    :param image_buffer_input: tensor corresponding to the image input
    :param keep_prob: tensor for controlling dropout rate
    :param prediction: tensor for the final prediction
    :return:
    """
    with tf.Session(graph=graph) as sess:
        feed_dict = {
            image_buffer_input: img,
            keep_prob: KEEP_ALL
        }

        prediction_result = sess.run([prediction], feed_dict)

    return int(prediction_result[0])


"""
===========================================================================
Audio processing
===========================================================================
"""


def _butter_bandpass_filter(samples, sample_rate, lowcut=30, highcut=3000, order=5):
    """
    Butterworth's filter
    """

    def butter_bandpass(lowcut, highcut, sample_rate, order=5):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = lfilter(b, a, samples)
    return y


def _get_melspectrogram(samples, sample_rate):
    """
    return a normalized spectrogram of type uint8

    INPUT
        samples
        sample_rate
    OUTPUT
        spectrogram   2D array, where axis 0 is time and axis 1 is fourier decomposition
                      of waveform at different times
    """
    melspectrogram = librosa.feature.melspectrogram(samples, sample_rate)

    # max L-infinity normalized the energy
    normalized = librosa.util.normalize(melspectrogram)

    # scale to 8-bit representation
    scaled = 255 * normalized  # Now scale by 255
    return scaled.astype(np.uint8)


def _repeat_channels(layer, dups=3):
    """
    Images used for training should contain 3 channels. This function
    duplicates the 2D array 3 times to conform to this requirement.
    :param layer:
    :param dups:
    :return:
    """
    return np.stack((layer for _ in range(dups)), axis=2)


def spectrogram(audio_file):
    samples, sample_rate = librosa.load(audio_file)
    samples = _butter_bandpass_filter(samples, sample_rate)

    return _repeat_channels(_get_melspectrogram(samples, sample_rate))


"""
===========================================================================
Run prediction
===========================================================================
"""
LABELS = dict(zip(range(8),
                  ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']))


def run_prediction(graph_file, audio_file):
    pred_index = predict(*get_tensors_for_prediction(graph_file),
                         spectrogram(audio_file))
    return LABELS[pred_index]


graph = '/Users/catherinehuang/Desktop/CS/ece496/tensorflow-for-poets-2/tf_files/aud_emo/retrained_graph.pb'
audio = '/Users/catherinehuang/Desktop/CS/ece496/audio_emotion_analysis/processed_at_lvl2/Audio_Speech_Actors_01-24/Actor_24/03-01-08-02-02-02-24-seg0.wav'
print(run_prediction(graph, audio))
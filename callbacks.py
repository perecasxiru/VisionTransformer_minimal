import math
import scipy
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from scipy import spatial
from keras.utils.data_utils import get_file
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import io
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class SimpleLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.train_writer = tf.summary.create_file_writer(log_dir+'/train')
        self.val_writer = tf.summary.create_file_writer(log_dir+'/val')
    
    def on_epoch_begin(self, epoch, logs=None):
        # Learning rate
        # try:
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr(self.model.optimizer.iterations)))        

        for writer in [self.train_writer, self.val_writer]:
            with writer.as_default():
                tf.summary.scalar('details/learning-rate', data=lr, step=epoch)        
                writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # Log metrics
        for k,v in logs.items():
            writer = self.val_writer if k.startswith('val_') else self.train_writer
            name = k[4:] if k.startswith('val_') else k
            with writer.as_default():
                tf.summary.scalar(name, data=v, step=epoch)
                writer.flush()

            
class ConfusionMatrixLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, val_generator, class_names, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.writer = tf.summary.create_file_writer(log_dir+'/val')
        self.val_generator = val_generator
        self.class_names = class_names

    def on_epoch_begin(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.val_generator)
        test_pred = np.argmax(test_pred_raw, axis=1)
        
        y_test = np.array([y for _, y in self.val_generator])
        test_labels = y_test.reshape(np.prod(y_test.shape[:2]), len(self.class_names)).argmax(1)

        # Calculate the confusion matrix.
        # cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
        
        fig, ax = plt.subplots(figsize=(10,10))
        ConfusionMatrixDisplay.from_predictions(test_labels, test_pred, 
                                                display_labels=self.class_names, ax=ax, 
                                                xticks_rotation='vertical', normalize='true')
        plt.tight_layout()
        
        # Log the confusion matrix as an image summary.
        # figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = plot_to_image(fig)

        # Log the confusion matrix as an image summary.
        with self.writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
            self.writer.flush()
            
            
class PositionEmbeddingLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.writer = tf.summary.create_file_writer(log_dir+'/val')

    def get_patch_attention(self):
        # Get the embedding layer weights
        emb = np.array(self.model.get_layer('Transformer/posembed_input').weights[-1])
        n = int(math.sqrt(len(emb)))

        sims = []
        for i in range(n*n):
            sim = [1-spatial.distance.cosine(emb[i+1], emb[j]) for j in range(1,len(emb))]
            sims.append(np.reshape(sim, (n,n)))

        return np.array(sims)


    def on_epoch_begin(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        
        sims = self.get_patch_attention()
        n = int(math.sqrt(len(sims)))

        fig, axs = plt.subplots(n,n,figsize=(10,10))
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            sim = sims[i]
            ax.set_axis_off()
            ax.imshow(np.reshape(sim,(n,n)))

        plt.tight_layout()
        
        # Log the as an image summary.
        patch_image = plot_to_image(fig)

        # Log it as an image summary.
        with self.writer.as_default():
            tf.summary.image("Patch Attention", patch_image, step=epoch)
            self.writer.flush()
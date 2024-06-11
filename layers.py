import math
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras.utils.data_utils import get_file
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

class MLP(Layer):
    def __init__(self, hidden_units, dropout_rate, is_trainable=True, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.is_trainable = is_trainable
        
    def build(self, input_shape):
        self.dense1 = Dense(self.hidden_units, activation=tf.nn.gelu, name=f"Dense_0", trainable=self.is_trainable)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense2 = Dense(input_shape[-1], activation='linear', name=f"Dense_1", trainable=self.is_trainable)
        self.dropout2 = Dropout(self.dropout_rate)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"hidden_units": self.hidden_units,
                       "dropout_rate": self.dropout_rate})
        return config
    
    
class Patches(Layer):
    def __init__(self, patch_size, projection_dim, is_trainable=True, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.is_trainable = is_trainable
        
    def build(self, input_shape):
        self.embedding = Conv2D(filters=self.projection_dim, kernel_size=self.patch_size, 
                                strides=self.patch_size, trainable=self.is_trainable, 
                                padding="valid", name='embedding')

    def call(self, images):        
        patches = self.embedding(images)        
        return patches

    def get_config(self):
        config = super().get_config()
        # config.update({"patch_size": self.patch_size})
        return config
    
    def compute_output_shape(self, input_shape):
        dim = input_shape[1] // self.patch_size
        return (input_shape[0], dim, dim, self.projection_dim)
    
    
class Patches3D(Layer):
    """
    Tubelet Embedding for the ViViT.
    """
    def __init__(self, num_images, patch_size, projection_dim, is_trainable=True, **kwargs):
        super(Patches3D, self).__init__(**kwargs)
        self.num_images = num_images
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.is_trainable = is_trainable
        
    def build(self, input_shape):
        self.embedding = Conv3D(filters=self.projection_dim, kernel_size=(self.num_images, self.patch_size, self.patch_size), 
                                strides=(self.num_images, self.patch_size, self.patch_size), trainable=self.is_trainable, 
                                padding="valid", name='embedding')

    def call(self, images):        
        patches = self.embedding(images)        
        return patches

    def get_config(self):
        config = super().get_config()
        return config
    
    
class ClassToken(Layer):
    def __init__(self, projection_dim, is_trainable=True, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.is_trainable = is_trainable
        
    def build(self, input_shape):
        self.token = tf.Variable(initial_value=tf.zeros_initializer()(shape=(1, 1, self.projection_dim), dtype="float32"), 
                                 trainable=self.is_trainable, name='cls')
    
    def call(self, patch):
        repeated_vec = tf.broadcast_to(self.token, (tf.shape(patch)[0], 1, self.projection_dim))
        encoded_patches = Concatenate(axis=1)([repeated_vec, patch])
        return encoded_patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"projection_dim": self.projection_dim})
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+1, input_shape[2])
    
    
# From https://github.com/faustomorales/vit-keras/blob/28815edc5c24492612af726d1b2ca78295128d84/vit_keras/layers.py#L66
class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = Dense(hidden_size, name="query")
        self.key_dense = Dense(hidden_size, name="key")
        self.value_dense = Dense(hidden_size, name="value")
        self.combine_heads = Dense(hidden_size, name="out")
        self.dropout = Dropout(self.dropout_rate)

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        
        output = self.combine_heads(concat_attention)
        output = self.dropout(output)
        
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, is_trainable=True, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.is_trainable = is_trainable
        
    def build(self, input_shape):
        self.position_embedding = Embedding(
            input_dim=self.num_patches+1, output_dim=self.projection_dim, 
            trainable = self.is_trainable,
            name='pos_embedding')
        
    def call(self, patch):
        # Initialize always the same vector
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)     # For the positional encoding
        
        # Retrieve the position embedding for each position
        pos_emb = self.position_embedding(positions)
        
        # Project the patch to match the positional embedding and add it
        encoded = patch + pos_emb
        return encoded, pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape, input_shape[1:])
    
    
# Position encoding using trigonometric functions
class FixedPatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(FixedPatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
    
    def get_position_encoding(self, n=10000):
        seq_len, d = self.num_patches+1, self.projection_dim
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
    
    def build(self, input_shape):
        self.position_embedding = Embedding(
            input_dim=self.num_patches+1, output_dim=self.projection_dim, 
            trainable = False,
            weights=[self.get_position_encoding()],
            name='pos_embedding')
        
    def call(self, patch):
        # Initialize always the same vector
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)     # For the positional encoding
        
        # Retrieve the position embedding for each position
        pos_emb = self.position_embedding(positions)
        
        # Project the patch to match the positional embedding and add it
        encoded = patch + pos_emb
        return encoded, pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
    
class StochasticDepth(Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=True):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"stoch_prob": self.drop_prob})
        return config
    
    
class TransformerBlock(Layer):
    def __init__(self, num_heads, mlp_head_units, dropout_rate, stoch_prob, is_trainable=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.mlp_head_units = mlp_head_units
        self.dropout_rate = dropout_rate
        self.stoch_prob = stoch_prob
        self.is_trainable = is_trainable
    
    def build(self, input_shape):
        self.layer_normalization1 = LayerNormalization(epsilon=1e-6, trainable=self.is_trainable, name="LayerNorm_0")
        self.layer_normalization2 = LayerNormalization(epsilon=1e-6, trainable=self.is_trainable, name="LayerNorm_2")
        self.attention = MultiHeadSelfAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate, name='MultiHeadDotProductAttention_1')
        
        self.mlp = MLP(hidden_units=self.mlp_head_units, dropout_rate=self.dropout_rate, trainable=self.is_trainable, name='MlpBlock_3')
        
        self.stoch_depth = StochasticDepth(self.stoch_prob)
        
        
    def call(self, encoded_patches, training=False):
        x1 = self.layer_normalization1(encoded_patches)
        
        # Create a multi-head attention layer.
        attention_output, attention_weights = self.attention(x1, training=training)
        attention_output = self.stoch_depth(attention_output) if self.stoch_prob>0 else attention_output
        
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        
        # Layer normalization 2.
        x3 =self.layer_normalization2(x2)
        
        # MLP.
        x3 = self.mlp(x3, training=training)
        x3 = self.stoch_depth(x3) if self.stoch_prob>0 else x3
        
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])
        
        return encoded_patches, attention_weights
    
    def get_config(self):
        config = super().get_config()
        return config
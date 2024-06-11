from layers import *
import tensorflow_addons as tfa

class VisionTransformer(tf.keras.models.Model):
    def __init__(self, input_shape, patch_size, num_patches, projection_dim, transformer_layers, num_heads, mlp_head_units, include_top, num_classes,
                 resize_to=None, top_activation='softmax', dropout_rate=0.1, stoch_prob=0):
        super().__init__()
        
        self._input_shape = input_shape
        
        self.resize_to = resize_to
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_head_units = mlp_head_units
        self.include_top = include_top
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.stoch_prob = stoch_prob        
        self.top_activation = top_activation
        
        self.patches = Patches(self.patch_size, self.projection_dim, name='patches')
        self.token = ClassToken(self.projection_dim, name='class_token')
        self.patch_encoder = PatchEncoder(self.num_patches, self.projection_dim, name='Transformer/posembed_input')
    
        self.transformer_blocks = [TransformerBlock(self.num_heads, self.mlp_head_units, dropout_rate=self.dropout_rate, stoch_prob=self.stoch_prob,
                                                    name=f"Transformer/encoderblock_{n}") for n in range(self.transformer_layers)]
        
        self.lay_norm = LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")
        self.head =  Dense(self.num_classes, activation=self.top_activation, name='head')
    
    def call(self, inputs, training=False):
        resized = Resizing(self.resize_to, self.resize_to)(inputs) if self.resize_to is not None else inputs
        
        # preprocess here
        
        patches = self.patches(resized)
        
        img_size = input_shape[0] if self.resize_to is None else self.resize_to
        dim = img_size // self.patch_size
        encoded_patches = Reshape((dim*dim,self.projection_dim))(patches)
        
        encoded_patches = self.token(encoded_patches)
        encoded_patches, pos_emb = self.patch_encoder(encoded_patches)
        
        for transf_block in self.transformer_blocks:
            encoded_patches, _ = transf_block(encoded_patches)
        
        representation = self.lay_norm(encoded_patches)
        tok = Lambda(lambda v: v[:, 0])(representation)
        
        if self.include_top:
            tok = self.head(tok)
        
        return tok
    
    def compile(self, num_classes, learning_rate=2e-4, weight_decay=0):
        
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0)
        
        return super().compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(name=f'loss'),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name=f'metrics/accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(3, name=f'metrics/top-3-accuracy'),
                tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='metrics/F1-macro'),
                tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes, name='metrics/AUC'),
                tf.keras.metrics.Precision(name='metrics/precision'),
                tf.keras.metrics.Recall(name='metrics/recall'),
                tf.keras.metrics.PrecisionAtRecall(0.99, name='metrics/P@R_99'),
                tf.keras.metrics.PrecisionAtRecall(0.95, name='metrics/P@R_95'),
                tf.keras.metrics.PrecisionAtRecall(0.9, name='metrics/P@R_90'),
                tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_classes, name='metrics/MCC')
            ],
        )
            
    def summary(self, line_length=120, expand_nested=False):
        x = Input(self._input_shape)
        dummy_model = Model(inputs=[x], outputs=self.call(x))
        dummy_model.summary(line_length=line_length, expand_nested=expand_nested)
        
    def plot(self, rankdir='TB'):
        x = Input(self._input_shape)
        dummy_model = Model(inputs=[x], outputs=self.call(x))
        return tf.keras.utils.plot_model(dummy_model, show_shapes=True, rankdir=rankdir)
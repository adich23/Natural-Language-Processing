import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GRU, Bidirectional, LSTM

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))


        ### TODO(Students) START
        # ...
        forward_layer = GRU(units=hidden_size, return_sequences=True)
        self.model = Bidirectional(forward_layer)

        ### TODO(Students) END




    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        M = tf.math.tanh(rnn_outputs)
        w_m = tf.tensordot(M, self.omegas, axes=[[2], [0]])
        alpha = tf.nn.softmax(w_m, axis=1)

        r = tf.multiply(rnn_outputs, alpha)

        intermediate = tf.reduce_sum(r, 1)

        return tf.tanh(intermediate)

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)


        ### TODO(Students) START
        # ...
        embedding_mask = inputs != 0

        concat_emb = tf.concat([word_embed,pos_embed],axis=2)

        first = self.model(concat_emb,mask=embedding_mask, training=training)
        # Best model is using only Word + Dependency structures
        # first = self.model(word_embed, mask=embedding_mask, training=training)
        attn_outputs = self.attn(first)

        logits = self.decoder(attn_outputs)

        ### TODO(Students) END

        return {'logits': logits}

class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, training: bool = False):
        super(MyAdvancedModel, self).__init__()

        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        self.drop1 = layers.Dropout(0.3)
        filters = embed_dim

        kernel_size_1 = 2 # words to convolute at once
        kernel_size_2 = 3
        self.conv1 = layers.Conv1D(filters, kernel_size_1)
        self.conv2 = layers.Conv1D(filters, kernel_size_2)

        self.drop2 = layers.Dropout(0.5)
        self.decoder = layers.Dense(units=self.num_classes)

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        concat_emb = word_embed
        if training:
            concat_emb = self.drop1(concat_emb)

        out_1 = self.conv1(concat_emb)
        out_2 = self.conv2(concat_emb)

        shape_1 = out_1.shape[1]
        pool1 = layers.MaxPool1D(shape_1)
        out_1 = tf.nn.relu(out_1)
        out_1 = pool1(out_1)
        out_1 = tf.squeeze(out_1)

        shape_2 = out_2.shape[1]
        pool2 = layers.MaxPool1D(shape_2)
        out_2 = tf.nn.relu(out_2)
        out_2 = pool2(out_2)
        out_2 = tf.squeeze(out_2)

        # batch_size * no_of_filters * 2
        concat_pool = tf.concat([out_1, out_2], axis=1)

        if training:
            concat_pool = self.drop2(concat_pool)
        logits = self.decoder(concat_pool)

        return {'logits': logits}

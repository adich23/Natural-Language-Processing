import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GRU, Bidirectional

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
        backward_layer = GRU(units=hidden_size,  return_sequences=True, go_backwards=True)
        self.model = Bidirectional(forward_layer, backward_layer=backward_layer,
                                input_shape=(None, 2*embed_dim))

        ### TODO(Students) END




    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        H = tf.transpose(rnn_outputs, perm=[0,2,1])
        M = tf.math.tanh(H)
        # (1*256) X (10,256,5) = (10,5) or (10,1,5)
        dot = tf.matmul(self.omegas,M,transpose_a=True)
        alpha = tf.nn.softmax(dot,axis=2)
        # (10, 256, 5)X(5,10)
        # dot = tf.tensordot(M, tf.transpose(self.omegas), axes=[1,1])
        # alpha = tf.squeeze(dot)
        # alpha = tf.nn.softmax(alpha,axis=1)
        # alpha = tf.expand_dims(alpha, axis=1)

        r = tf.matmul(H,alpha,transpose_b=True)
        r = tf.squeeze(r)
        output = tf.math.tanh(r)

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)


        ### TODO(Students) START
        # ...
        '''
           dropout embedding
           layer, dropout LSTM layer and dropout the penultimate layer,
           dropout rate is set as 0.3, 0.3, 0.5
       '''
        # embedding_mask  = tf.cast(inputs!=0, tf.float32)

        embedding_mask = inputs != 0

        # embedding_mask = tf.concat([embedding_mask,embedding_mask],1)
        # last dim should be 200
        concat_emb = tf.concat([word_embed,pos_embed],axis=2)

        # TODO merged_mode = 'sum'
        # should have 128*2 size in last layer
        first = self.model(concat_emb,mask=embedding_mask)

        attn_outputs = self.attn(first)

        logits = self.decoder(attn_outputs)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size , 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...

        forward_layer = GRU(units=hidden_size, return_sequences=True)
        backward_layer = GRU(units=hidden_size, return_sequences=True, go_backwards=True)
        self.model = Bidirectional(forward_layer, backward_layer=backward_layer,
                                   input_shape=(None, 2 * embed_dim))

        forward_layer1 = GRU(units=int(hidden_size/2), return_sequences=True)
        backward_layer1 = GRU(units=int(hidden_size/2), return_sequences=True, go_backwards=True)
        self.model1 = Bidirectional(forward_layer1, backward_layer=backward_layer1,
                                   input_shape=(None, 2*hidden_size))



        # ...
        ### TODO(Students END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        H = tf.transpose(rnn_outputs, perm=[0, 2, 1])
        M = tf.math.tanh(H)
        # (1*256) X (10,256,5) = (10,5) or (10,1,5)
        dot = tf.matmul(self.omegas, M, transpose_a=True)
        alpha = tf.nn.softmax(dot, axis=2)
        # (10, 256, 5)X(5,10)
        # dot = tf.tensordot(M, tf.transpose(self.omegas), axes=[1,1])
        # alpha = tf.squeeze(dot)
        # alpha = tf.nn.softmax(alpha,axis=1)
        # alpha = tf.expand_dims(alpha, axis=1)

        r = tf.matmul(H, alpha, transpose_b=True)
        r = tf.squeeze(r)
        output = tf.math.tanh(r)

        ### TODO(Students) END

        return output




    def call(self, inputs, pos_inputs, training):
        # raise NotImplementedError
        ### TODO(Students) START

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        embedding_mask = inputs != 0

        # embedding_mask = tf.concat([embedding_mask,embedding_mask],1)
        # last dim should be 200
        concat_emb = tf.concat([word_embed, pos_embed], axis=2)

        # TODO merged_mode = 'sum'
        # should have 128*2 size in last layer
        first = self.model(concat_emb, mask=embedding_mask)

        second = self.model1(first)
        attn_outputs = self.attn(second)
        # attn_outputs = second
        logits = self.decoder(attn_outputs)

        ### TODO(Students) END

        return {'logits': logits}
# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self._num_layers = num_layers
        self._dropout = dropout
        self.dense = []
        for i in range(num_layers):
            self.dense.append(tf.keras.layers.Dense(input_dim, activation=tf.nn.relu))

        # self.dense5 = tf.keras.layers.Dense(input_dim, activation=tf.nn.softmax)

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        batch_size = vector_sequence.shape[0]
        max_tokens_num = vector_sequence.shape[1]
        embedding_dim = vector_sequence.shape[2]

        total_embeds = []
        for sentence in range(batch_size):
            selected_words = []
            layers = []
            # do dropout only when training
            mask = sequence_mask[sentence] > 0


            selected_words = tf.boolean_mask(vector_sequence[sentence],mask)
            if selected_words.shape[0] <= 0:
                selected_words = vector_sequence[sentence]

            if training:
                proba = tf.random.uniform(shape=[1,selected_words.shape[0]],minval=0, maxval=1,seed=None)
                mask = proba[0] > self._dropout
                selected_words = tf.boolean_mask(selected_words, mask)


            avg_tensor = tf.reduce_mean(selected_words, axis=0)
            comb_embeddings = tf.reshape(avg_tensor,[1,embedding_dim])
            total_embeds.append(comb_embeddings)
            # print ("comb embeddings",comb_embeddings.shape)

        total_embeds = tf.convert_to_tensor(total_embeds)
        layers.append(tf.reshape(self.dense[0](total_embeds),[batch_size,embedding_dim]))
        # TODO Relu vs Tanh

        for i in range(1,self._num_layers):
            # apply function on previous layer, store it.
            layers.append(self.dense[i](layers[i-1]))

        layers = tf.convert_to_tensor(layers)
        final_softmax = layers[-1]
        # final_softmax = self.dense5(layers[-1])
        # final_softmax = tf.reshape(final_softmax,[batch_size,embedding_dim])
        # TODO(students): end
        return {"combined_vector": final_softmax,
                "layer_representations": layers}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._num_layers = num_layers
        self.grus = []
        for i in range(num_layers):
            self.grus.append(tf.keras.layers.GRU(input_dim,  return_sequences=True))

        # ...
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        #
        layers = []
        sequence_mask = tf.dtypes.cast(sequence_mask, tf.bool)
        layers.append(self.grus[0](vector_sequence,mask=sequence_mask))

        for i in range(1, self._num_layers):
            # apply function on previous layer, store it.
            layers.append(self.grus[i](layers[i - 1]))

        # TODO verify this logic
        # combined_vector = tf.convert_to_tensor(layers[-1][:,-1,:])
        combined_vector = layers[-1][:,-1,:]
        last_layers = [layer[:,-1,:] for layer in layers]
        last_layers = tf.convert_to_tensor(last_layers)
        # ...
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": last_layers}

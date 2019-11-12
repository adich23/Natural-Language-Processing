# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        # raise NotImplementedError
        return tf.math.pow(vector,3)
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self._hidden_dim = hidden_dim
        # self.hidden_layer = tf.keras.layers.Dense((self._hidden_dim, int(embedding_dim) * num_tokens),
        #                                           input_shape=(int(embedding_dim) * num_tokens,),
        #                                           activation=self._activation)
        #
        # self.output_layer = tf.keras.layers.Dense((self._hidden_dim, num_transitions))

        self._trainiable = trainable_embeddings

        # defining layer architecture
        emb_size = int(embedding_dim) * num_tokens

        w_init = tf.random.truncated_normal

        self.w1 = tf.Variable(initial_value=w_init([emb_size, self._hidden_dim],
                                                   stddev=1.0 / math.sqrt(emb_size)), trainable=self._trainiable)

        b_init = tf.zeros_initializer()
        self.b1 = tf.Variable(initial_value=b_init(shape=(self._hidden_dim,)), trainable=self._trainiable)

        self.w2 = tf.Variable(initial_value=w_init([self._hidden_dim, num_transitions],
                                                   stddev=1.0 / math.sqrt(self._hidden_dim)),
                              trainable=self._trainiable)
        # self.b2 = tf.Variable(initial_value=b_init(shape=(num_transitions,), trainable=self._trainiable))

        # TODO trainable in embeddings init
        self.embeddings = tf.Variable(initial_value=tf.random.uniform([vocab_size, embedding_dim], -0.01, 0.01)
                                      , trainable=self._trainiable)
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        emb_inputs = tf.reshape(tf.nn.embedding_lookup(self.embeddings,inputs),[inputs.shape[0],self.w1.shape[0]])
        imdt_1 = tf.matmul(emb_inputs, self.w1) + self.b1
        op_1 = self._activation(imdt_1)

        #TODO verify op1 shape = 10k * self._hidden_dim

        imdt_2 = tf.matmul(op_1,self.w2) #+ self.b2
        logits = imdt_2
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def stable_softmax(self,logits):
        scaled_logits = logits
        numer = tf.where(scaled_logits == 0.0 ,0,tf.exp(scaled_logits))
        numer = tf.where(numer == -0.0, 0, numer)
        softmax = numer/tf.reduce_sum(numer,1,keepdims=True)
        return softmax

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        # we compute the softmax probabilities only among the feasible transitions in practice.
        # This means you want to use labels to mask out the infeasible moves,
        # so that you do not include them in your calculations.

        # NOTE softmax probabilities of correct transition among the feasible transitions.
        # means who has label = 1
        # TODO label 0 and 1
        mask = labels > -1
        # filtered_logits = tf.boolean_mask(logits, mask)
        filtered_logits = tf.multiply(logits,mask)

        sf = self.stable_softmax(filtered_logits)

        #TODO label 1
        eps = 1e-10
        mask_2 = labels > 0
        # In Cross entropy loss[ y_i* log(p_i) ] only taking correct ones , therefore y_i = 1
        p_vec = tf.math.log(sf+eps)
        p_i = tf.multiply(p_vec,mask_2)
        loss = -tf.reduce_sum(p_i,axis=None)
        loss/= logits.shape[0]

        regularization = 0
        if self._trainiable:
            # theta = self.w1 + self.b1 + self.w2 + self.embeddings
            regularization += tf.nn.l2_loss(self.w1)
            regularization += tf.nn.l2_loss(self.b1)
            regularization += tf.nn.l2_loss(self.w2)
            regularization += tf.nn.l2_loss(self.embeddings)
            regularization*= self._regularization_lambda
        # TODO(Students) End
        return loss + regularization

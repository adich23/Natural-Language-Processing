## Results Analysis

<img src="https://github.com/adich23/Natural-Language-Processing/blob/master/Dependency%20Parsing/results/results-1.png" width="50%">

1. Cubic activation function outperforms the other two. Performance of the activation functions for
Dependency parsing task can be written as - **Cubic > Tanh > Sigmoid**.
2. Cubic function gives 0.5 - 1.8% improvement over tanh and sigmoid function in
UAS metrics.
3. Pretained embeddings gives around 2% improvement over training embeddings from scratch in UAS and LAS metrics.
4. This is coherent with what was analysed in the research paper.
5. When we trained embeddings from scratch it achieves comparable accuracy across all the metrics.
6. Accuracy decreasses in every metrics when embeddings are kept frozen. This is
because the model is not allowed to learn the word relations and context, specific
to this dataset. Glove embeddings are trained on different task and dataset and
generally cannot represent(to 100 %), every type of downstream dataset it is
used for.
<!--
|             |   UAS  | UAS no Punc |   LAS  | LAS no Punc |   UEM  | UEM no Punc |  Root  |
|:-----------:|:------:|:-----------:|:------:|:-----------:|:------:|:-----------:|:------:|
|    Cubic    | 87.972 |    89.642   | 85.405 |    86.734   | 35.411 |    38.529   | 89.941 |
|   Sigmoid   | 86.113 |    87.896   | 83.647 |    85.104   | 30.058 |    32.705   | 87.352 |
|     Tanh    | 87.424 |    89.043   | 85.086 |    86.395   | 34.294 |    36.882   | 88.176 |
|   Wo_glove  | 85.968 |    87.794   | 83.418 |    84.917   | 30.764 |    33.352   | 85.823 |
| Wo_emb_tune | 84.049 |    85.881   | 81.451 |    82.959   |  27.0  |    29.176   | 82.588 |
-->

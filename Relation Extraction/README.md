### Advanced Model
* This model is motivated from this reference paper - https://www.aclweb.org/anthology/C14-1220.pdf
* My best model achieved validation F-1 score 0.6359.
* I used *Convolution* layer in the beginning to extract high level features from the input matrix. Applying different kernel_size Convolution layers gives us varied size *'n-gram'* features from the input text. Since each sentence contained max. 5 words, I kept the kernel_size below it.
* Then *pooling* layer is applied to abstract the features generated from the convolutional layer by aggregating the scores for each filter while preserving the relative positions of the 'n-grams' between themselves and the entity heads at the same time.
* *MaxPool* is applied since it identifies the most important or relevant features from the sequence.
* *Dropout* is applied at two instances for regularization of model. First at the input feature layer and second while merging the sequences from Convolution layer. Experimented with values from [0.2,0.5] and selected best performing ones from validation set loss score.
* Tried with tanh and relu activation functions, where relu was performing better. Also experimented with different inputs to the model, in which 'Word+dependency structure' performed best.


### Advanced Model Architecture
![Model Architecture](https://github.com/adich23/Natural-Language-Processing/blob/master/Relation%20Extraction/data/Architecture.png)

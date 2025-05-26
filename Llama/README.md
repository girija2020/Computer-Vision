# Llama: Open and Efficient Foundational Language Model

*Hugo et al.*

Chinchilla Scaling Law - The best models are the ones that are trained on a large data and not those that are huge in sizes.

### Data collection

Large scale data collection and deduplication from a variety of sources ranging from line matches to exact file matches. Some data has been retained mainly based on punctuations and number of words in a sentence or sentences in a file.

### Architectural Changes

__Pre-normalization__: Normalizing the input to each sub layer of the transformer rather than the output. RMS Norm was used to improve training stability

Layer Normalization does the following:

1) Internal covariate shift issue is tackled, where a layer’s input distribution changes as previous layers are updated, which significantly slows the training
   $ a_i = \Sigma_{j} w_{ij}x_j $
   $ a_i = (a_i - \mu)/\sigma $
2) Normalizes the activations of the previous layer for each given example in a batch independently, rather than across a batch like Batch Normalization. i.e. applies a transformation that maintains the mean activation within each example close to 0 and the activation standard deviation close to 1. But Layer Normalization is expensive to compute and hence the RMS Norm is introduced.
3) A well-known explanation of the success of LayerNorm is its re-centering and re-scaling invariance property. The former enables the model to be insensitive to shift noises on both inputs and weights, and the latter keeps the output representations intact when both inputs and weights are randomly scaled. In this paper, we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance.
=> RMSNorm = $ a_i = a_i/RMS(a) $
=> RMS(a) = $ \sqrt{\Sigma a_i^2/n} $

SwiGLU activation function: SwiGLU activation function was used instead of RELU activation function with 2/3*4d dimension

Swish Activation Function: $ Swish_{\beta}(x) =  x * \sigma(\beta x) $
GLU: $ GLU(x, W, V, b, c) = \sigma(xW + b) * (xV + c) $
SwiGLU: $ SwiGLU(x, W, V, b, c, \beta) = Swish_{\beta}(xW + b) * (xV + c) $

This works lol! In author's own words:
"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

Rotary Position Embeddings: Instead of fixed position embeddings, rotary position embeddings are added

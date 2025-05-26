# Auto Regressive Image Generation

The success of Stable Diffusion was attributed to the UNet backbone which preserves the inductive bias on visual signals. This was to see if without the inductive bias, next token prediction would be able to generate images similar to the Llama next word prediction.

1) Well designed Image Compressors
2) Scalable Image Gen Models
3) High Quality training Data

Let $ x \in R^{H*W*3} $ the we have discrete image tokens $ q \in Q^{h*w} $ where $ h = H/p $ and $ w = W/p $ where p is the scaling factor and $ q^{(i,j)} $ is indices of image in the codebook.
Codebook
Raster Scan Ordering

### Image tokenizer

### Class conditional Image Generation Models

### Text Conditional Image Generation Models

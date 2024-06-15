# nanodecoder

A simple implementation of a Transformer decoder (GPT) that will *hopefully* include the following:
- [SwiGLU](https://arxiv.org/abs/2002.05202): instead of ReLU / GELU as an activation function in MLPs
- [RoPE](https://arxiv.org/abs/2104.09864): encode position with rotation matrix
- [Partial rotary embedding](https://huggingface.co/microsoft/phi-2/blob/c929c7735ac31aa03ef9a1e8d72c5d2f62999e27/modeling_phi.py#L343): apply RoPE only to part of q, v vectors
- [Pre-LN transformer](https://arxiv.org/abs/2002.04745): layer normalization placed in residual blocks
- [RMSNorm](https://arxiv.org/abs/1910.07467): regularize neuron inputs in a layer using root mean square
- [Mutli-head latent attention](https://arxiv.org/abs/2405.04434): compress key, values for inference efficiency
- [Sparse mixtrue of experts](https://arxiv.org/abs/2209.01667): multiple MLPs and a router that dynamically routes input tokens to specific experts for processing

> Inspired by Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT)

# THESYNERGY OFUNIVERSALTRANSFORMERS ANDROTARY

# EMBEDDINGS

#### A PREPRINT

```
Eric Houzelle
Independent Researcher
eric@hektore.com
```
```
December 2025
```
## ABSTRACT

```
Parameter redundancy remains a significant bottleneck for deploying Large Language Models (LLMs)
on resource-constrained devices. This paper investigates the effectiveness ofSuper Tiny Lan-
guage Models (STLMs)that prioritize extreme parameter efficiency over raw scale. We explore a
Transformer architecture utilizing aggressive cross-layer weight sharing—inspired by the Universal
Transformer—augmented with Rotary Positional Embeddings (RoPE) to decouple sequence depth
from positional encoding. Evaluated on a French adaptation of theTinyStoriesdataset, our results
demonstrate that a fully weight-shared model achieves a34% to 60% reduction in parameters
while maintaining, and occasionally improving, validation perplexity compared to standard baselines.
These findings suggest that the combination of recurrent depth and modern embeddings offers a
sustainable path for edge-compatible language modeling.
```
## 1 Introduction

The dominant paradigm in large language model research follows empirical scaling laws [ 1 ], where performance
improves monotonically with increases in parameter count, data size, and compute. While successful, this paradigm
neglects two increasingly important constraints:

1. Edge and Mobile Deployment: Many real-world applications target mobile phones or embedded devices
    with strict memory budgets, often below 4 GB of VRAM.
2. Low-Resource Languages: For many languages, training data is limited, making large models prone to
    overfitting and inefficient use of parameters.

In these regimes,parameter efficiencymay be more valuable than sheer scale. This has led to the emergence ofSuper
Tiny Language Models (STLMs), a category of models specifically optimized for sub-100M parameter counts [ 6 ].
Motivated by this, we revisit theUniversal Transformerarchitecture [ 2 ], which enforces cross-layer weight sharing,
effectively transforming the Transformer into a recurrent computation in depth. While previous work has shown mixed
results for open-ended generation, we hypothesize that modern positional encodings can significantly mitigate these
issues.

We combine weight sharing withRotary Positional Embeddings (RoPE)[ 4 ], which encode relative positions inde-
pendently of layer index. This property is particularly well-suited to shared-weight architectures, where learned absolute
positional embeddings can conflict with layer reuse.

Our contributions are:

- An empirical study of fully and partially shared Transformer architectures in the ultra-small regime (8M–45M
    parameters).


### The Synergy of Universal Transformers and Rotary Embeddings A PREPRINT

- Evidence that full weight sharing can outperform standard Transformers at small scale and remain competitive
    at larger depths.
- A case study on French TinyStories, demonstrating that efficient architectures are well-suited for low-resource
    and edge scenarios.

## 2 Related Work

Universal Transformers[ 2 ] introduced the idea of tying parameters across layers, showing improved generalization
on algorithmic tasks but reporting optimization difficulties in generative settings.

ALBERT[ 3 ] demonstrated that cross-layer parameter sharing can act as a strong regularizer in BERT-style models,
significantly reducing parameter count with minimal loss in accuracy.

TinyStories[ 5 ] showed that simplified language data allows small models to learn coherent reasoning and narrative
structure, making it an ideal testbed for architectural experiments.

Recently,Hillier et al. (2024)[ 6 ] demonstrated that STLMs can achieve significant compression through byte-level
tokenization and weight tying. While their work focuses on tokenization strategies, our approach investigates the
architectural synergy between full parameter sharing and Rotary Positional Embeddings (RoPE). RoPE [ 4 ] is particularly
crucial here as it replaces learned absolute embeddings with rotation matrices, removing the dependency between
positional encoding and layer depth and potentially solving stability issues in recurrent Transformer architectures.

## 3 Methodology

3.1 STLM Architecture

Our Super Tiny Language Model (STLM) is a causal, decoder-only Transformer. Letxldenote the hidden state at layer
l. A standard Transformer blockFθlis defined as:

```
xl+1=Fθl(xl),
```
where each layer has its own parametersθl.

In theUniversal (Fully Shared)variant, we enforce:

```
θl=θ ∀l,
```
yielding:
xl+1=Fθ(xl).

This formulation increases model depth by adding computation (FLOPs) without increasing parameter memory. While
this recurrence resembles an RNN, it is important to note that recurrence occursin depth rather than in time: the full
sequence is processed at each step, preserving Transformer-style parallelism over tokens.

3.2 Variants Evaluated

We evaluate three architectures:

1. Baseline: Standard GPT-style Transformer with unique attention and feed-forward weights.
2. STLM-FFN: Shared feed-forward network (FFN) weights across layers, with unique attention parameters
    (approximately 20–25% reduction).
3. STLM-Universal: All Transformer block parameters shared across layers (34–60% reduction depending on
    depth).

All models use RoPE for positional encoding.

## 4 Experimental Setup

Dataset. We useTinyStories-French, obtained by machine-translating the original English TinyStories dataset using a
high-quality neural translation system. No post-editing was performed. While translation noise is present, it affects all
models equally and does not favor shared-weight architectures.


### The Synergy of Universal Transformers and Rotary Embeddings A PREPRINT

Training. All models were trained with identical optimization settings, random seeds, and token budgets. We use
AdamW with a learning rate of 3 × 10 −^4 and a OneCycleLR schedule. Smaller models were trained for 10 epochs,
larger models for 2 epochs, ensuring comparable total token exposure.

Metrics. We report validation cross-entropy loss, perplexity, parameter count, and peak VRAM usage.

## 5 Results and Analysis

5.1 Small-Scale Efficiency (Depth=8, Embed=256)

```
Model Parameters Loss VRAM (Peak)
Baseline 13.45M 3.155 5.2 GB
STLM-FFN 10.70M (-20%) 3.145 5.2 GB
STLM-Universal 8.85M (-34%) 3.139 5.2 GB
```
Table 1: Results at small scale. Peak VRAM is dominated by activations rather than parameters and therefore remains
constant across variants.

At this scale, full weight sharing acts as a strong regularizer, allowing the Universal model to outperform the baseline
despite having 4.6M fewer parameters.

5.2 Scaling to Deeper Models (Depth=16, Embed=512)

```
Model Parameters Loss Reduction
Baseline 45.8M 3.177 0%
STLM-FFN 34.0M 3.171 -25%
STLM-Universal 18.2M 3.180 -60%
Table 2: Results at larger depth. Parameter count remains constant for the Universal model as depth increases.
```
At this scale, partial sharing (STLM-FFN) achieves the best absolute performance, while the Universal model remains
highly competitive despite being 2.5×smaller in parameter count.

5.3 Memory and Throughput Considerations

While weight sharing significantly reduces model storage and checkpoint size, peak VRAM usage and inference latency
remain similar across variants. This is expected, as activation memory and FLOPs scale with depth, not parameter
count. The primary advantages of shared-weight models are reduced storage, improved cache locality, and suitability
for deployment on memory-constrained devices.

5.4 Limitations

While our results are promising, several limitations should be noted:

1. The use of machine-translated data for the French TinyStories dataset may introduce translation artifacts;
2. The findings are currently limited to the simplified narrative domain of TinyStories and may not immediately
    generalize to complex factual reasoning;
3. Although weight sharing reduces storage and memory footprint, it does not reduce the total FLOPs required
    for inference, as the number of computational passes remains tied to the model depth.

## 6 Discussion

The success of Universal Transformers in this setting contrasts with earlier reports of convergence difficulties. We
hypothesize that two factors contribute to this stability: (1) RoPE decouples positional information from layer index,
avoiding conflicts under weight reuse; and (2) the simplified structure of TinyStories reduces long-range compositional
complexity, making recurrent depth refinement effective.


### The Synergy of Universal Transformers and Rotary Embeddings A PREPRINT

## 7 Conclusion

This work provides empirical evidence that Transformer parameter redundancy is substantial, particularly in low-
resource regimes. On French TinyStories, we show that:

- At very small scale (8M–13M parameters), full weight sharing outperforms standard Transformers.
- At small-to-medium scale (18M–45M parameters), partial sharing offers the best accuracy, while full sharing
    achieves up to 60% compression with negligible loss.

These findings support the use of shared-weight Transformers for Green AI and edge deployment. Future work will
exploreknowledge distillation, using Shared-FFN models as teachers for Universal students, to further close the
remaining performance gap at scale.

## References

[1] Jared Kaplan et al. Scaling laws for neural language models.arXiv preprint arXiv:2001.08361, 2020.

[2] Mostafa Dehghani et al. Universal Transformers.ICLR, 2019.

[3] Zhenzhong Lan et al. ALBERT: A Lite BERT.ICLR, 2020.

[4]Jianlin Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv preprint
arXiv:2104.09864, 2021.

[5]Ronen Eldan and Yuanzhi Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English?
arXiv preprint arXiv:2305.07759, 2023.

[6] Dylan Hillier, Leon Guertler, et al. Super Tiny Language Models.arXiv preprint arXiv:2405.14159, 2024.



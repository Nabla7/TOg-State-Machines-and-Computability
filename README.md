# TOg-State-Machines-and-Computability

=== Scratchpad :

Main paper : On the Turing Completeness of Modern Neural Network Architectures ( https://openreview.net/pdf?id=HyGBdo0qFm )

The reviews are invaluable to direct our methodology ( https://openreview.net/forum?id=HyGBdo0qFm )

---

### Interesting criticism is levied :

---

Potentially interesting results, very dense and confusing writing


The paper shows Turing completeness of two modern neural architectures, the Transformer and the Neural GPU. The paper is technically very heavy and gives very little insight and intuition behind the results. Right after surveying the previous work the paper starts stacking definitions and theorems without much explanations.

While technical results are potentially quite strong I believe a major revision to the paper might be necessary in order to clarify the ideas. I would even suggest to split the paper into two, one about each architecture as in the current form it is quite long and difficult to follow. 


--- 

### More critcism with regards to the veracity of the result, interesting thread:

---

- rewiever:
It's observed in [1] that Transformer is not universal. Also, the proofs in this paper are very technical without any intuitive explanation. The results seem very questionable. It is definitely necessary to address this concern before this paper can be accepted.


- authors:
We believe that your doubt has been already clarified by the authors of the paper mentioned in your comment ("Universal Transformers"), and we thank the authors for their response. We just want to emphasize that our results only hold when unbounded precision is admitted, which is a standard assumption in the theoretical analysis of the computational power of neural networks (see, e.g., the Universal  Approximation Theorem, or Turing Completeness for RNNs). As mentioned in the response provided by the authors of the Universal Transformers paper, when only bounded precision is allowed, then the model is no longer Turing complete. In fact, we formally prove in our paper that the latter holds even if one sees the Transformer as a seq-to-seq network that produces an arbitrary long output.  We will include some further comments about this in the final version of our paper.


- authors of [1] chime in:
Comment: We are the authors of the Universal Transformer paper ([1] above). As this comment is very similar to what was posted on that submission, please see our response there:  /forum?id=HyzdRiR9Y7&noteId=HyxfZDmCk4&noteId=rkginvfklN
The TLDR is that in this work the authors assume arbitrary-precision arithmetic, whereas in our case we focus on the fixed-precision setting and provide a fairly short and intuitive counterexample showing that the Transformer is not universal in that setting, whereas the Universal Transformer is (see our comment above). Our main focus in that work, however, is to show how this increased theoretical capacity leads to significant practical advantages by expanding the number of tasks the Transformer can solve, and by improving accuracies on multiple real-world sequence-to-sequence learning tasks such as MT.

---

Responses from the authors : 

- [comment] “Results are claimed to hold without access to external memory [...] what if the problem at hand is, say EXPSPACE-complete? Then the network would have to be of exponential size [...] The whole point of Turing-completeness is that the program size is independent of the input size so there seems to be some confusion here.”

> [response] As stated in the paper, Turing completeness for Transformer and Neural GPU is obtained by taking advantage of the internal representations used by both architectures. We prove that the Transformer and the Neural GPU can use the values in their internal activations to carry out the computations while having a network with a fixed number of neurons and connections. For the case of Neural GPUs we even restrict the architecture to ensure a fixed number of parameters (Uniform Neural GPUs). Thus our proof actually uses a “program size which is independent of the input size” as mentioned by the reviewer. The confusion might arise because of our assumption that internal representations are rational numbers with arbitrary precision; we are trading external memory by internal precision. This is a classical assumption in the study of the computational power of neural networks (e.g. Universal Approximation Theorem for FFNs and Turing Completeness for RNNs). We mention this property in the Introduction, in the Conclusions, and also when formally proving the results, but we will make it more explicit in the next version of the paper.

- [comment] “The paper is technically very heavy [...] I believe a major revision to the paper might be necessary in order to clarify the ideas.”

> [response] It is true that the paper is a bit dense, but we prove a technically involved result. To be precise in our claims we needed to include all the definitions in the paper. Moreover, our formal definitions can be used in the future to prove more properties for these and similar architectures with theoretical and practical implications. Though technical, the two other reviewers explicitly mention that the paper is well written. 

- [comment] “The paper [...] gives very little insight and intuition behind the results.”

> [response] The main intuition in our results is that both architectures can effectively simulate an (Elman)RNN-seq2seq computation, which by Siegelmann and Sontag’s classical result [1] are Turing complete when internal representations are rational numbers of arbitrary precision. We mentioned this in the Introduction and in each proof sketch, but we will make it more explicit in the next version of the paper.

- [comment] “I would even suggest to split the paper into two, one about each architecture”.

> [response] We wanted to have both architectures in the paper as they are two of the most popular architectures in use today, yet based on different paradigms; namely, self-attention mechanisms and convolution. We wanted to understand to what extent the use of these features could be exploited in order to show Turing completeness for the models. Moreover, the computational power of Transformers has been compared with that of Neural GPUs in the current literature, but both are only informally used. We wanted to provide a formal way of approaching this comparison.

#### The main assumption behind the conclusion really:

Our proofs are based on having unbounded precision for internal representations (neuron values). For weights one can prove that fixed precision (actually very small) is enough.

Our results say nothing about the computational power when fixed precision (like float32) is assumed for internal representations. We actually state the fixed-precision case as an interesting topic for future research.


### Initial thoughts :


> Results are interesting but there are some people who feel like the explanation is not clear or not well motivated. An interesting fact is the tradeof between external memory and precision : "we are trading external memory by internal precision" .

#### Questions
1. How can we motivate this assumption of infinite precision ?
2. What does it mean to make the trade of between external memory and internal precision ?
3. How can the answers to 1 and 2 lead to a better explanation of the proof in the paper that transformers are Turing Complete ?
4. Maybe we can build little transformers with increasing amounts of precision and perform tests ? Idk if thats interesting 


### Further background, this paper cites Jorge's paper: Looped Transformers as Programmable Computers (https://arxiv.org/pdf/2301.13196.pdf)

> Surprisingly, through in-context learning LLMs can perform algorithmic tasks and reasoning,
as demonstrated in several works including Nye et al. [2021], Wei et al. [2022c], Lewkowycz
et al. [2022], Wei et al. [2022b], Zhou et al. [2022], Dasgupta et al. [2022], Chung et al. [2022].
For example, Zhou et al. [2022] showed that LLMs can successfully perform addition on unseen
examples when prompted with a multidigit addition algorithm and a few examples of addition.
These results suggest that LLMs can apply algorithmic principles and perform pre-instructed
commands on a given input at inference time, as if interpreting natural language as code.
Constructive arguments have demonstrated that Transformers can simulate Turing Machines
with enough depth or recursive links between attention layers Pérez et al. [2021], Pérez et al. [2019],
Wei et al. [2022a]. This demonstrates the potential of transformer networks to precisely follow
algorithmic instructions specified by the input. Yet, these constructions are more generalized and
do not provide insight into how to create Transformers that can carry out particular algorithmic
tasks, or compile programs in a higher-level programming language

#### I'm trying to use this to guide my explanations of the proof with better link to implementation
Role of Positional Encodings and Internal Representations: A crucial aspect of the paper is how positional encodings and the model’s ability to compute and access internal dense representations of data enable Transformers to achieve Turing completeness. This is significant because it shows that Transformers can inherently process sequential data in a way that’s comparable to the sequential processing of a Turing machine.

-> https://aclanthology.org/2020.conll-1.37.pdf 

> We provide an alternate and arguably simpler proof to show that Transformers are Turingcomplete by directly relating them to RNNs.
> More importantly, we prove that Transformers with positional masking and without positional encoding are also Turing-complete.

 - In the limit of precision a fixed parameter transformer is Turing complete.

- perez is cited here https://arxiv.org/abs/1906.06755 Theoretical Limitations of Self-Attention in Neural Sequence Models:
  - In this work, we mathematically investigate the computational power of self-attention to model formal languages. Across both soft and hard attention, we show strong theoretical limitations of the computational abilities of selfattention, finding that it cannot model periodic finite-state languages, nor hierarchical structure, unless the number of layers or heads increases with input length. These limitations seem surprising given the practical success of self-attention and the prominent role assigned to hierarchical structure in linguistics, suggesting that natural language can be approximated well with models that are too weak for the formal languages typically assumed in theoretical linguistics.
  - > Theoretical study of transformers was initiate by Perez et al. (2019), who theoretically studied the ability of Seq2Seq transformers to emulate the computation of Turing machines.
   - > A related, though different, strand of research has investigated the power of neural networks to model Turing machines. A classical result (Siegelman and Sontag, 1995) states that—given unlimited computation time—recurrent networks can emulate the computation of Turing machines. Very recently, Perez et al. (2019) have shown the same result for both (argmax-attention) Transformers and Neural GPUs. The crucial difference between these studies and studies of language recognition is that, in these studies, the networks are allowed to perform unbounded recurrent computations, arbitrarily longer than the input length.
    - > Hard and Soft Attention There is a choice between soft attention and hard attention (Shen et al., 2018b; Perez et al., 2019). The one prior ´ theoretical study of transformers (Perez et al., ´ 2019) assumes hard attention. In practice, soft attention is easier to train with gradient descent; however, analysis studies suggest that attention often concentrates on one or a few positions in trained transformer models (Voita et al., 2019; Clark et al., 2019) and that the most important heads are those that clearly focus on a few positions (Voita et al., 2019), suggesting that attention often behaves like hard attention in practice. We will examine both hard (Section 5) and soft (Section 6) attention.
   - ### What:
      - We formally investigated the capabilities of selfattention in modeling regular languages and hierarchical structure. We showed that transformers cannot model periodic regular languages or basic recursion, either with hard or soft attention, and even if infinite precision is allowed. This entails that self-attention cannot in general emulate stacks or general finite-state automata. Our results theoretically confirm the idea that self-attention, by avoiding recurrence, has quite limited computational power.
      - ### **The crucial difference between these studies and studies of language recognition is that, in these studies, the networks are allowed to perform unbounded recurrent computations, arbitrarily longer than the input length.**
    - ### How it relates to the arguments by Perez et al.:
       - In the case of transformers, while Hahn's paper points out their limitations in emulating stacks and thereby their difficulties with certain structured tasks, this does not necessarily negate their Turing completeness. Transformers, especially with certain configurations or enhancements, can simulate the basic operations of a Turing machine. Their limitation in handling certain language structures more effectively points to a difference in efficiency or suitability for specific tasks, rather than a fundamental limitation in their computational power.

### Summary again :

- Computational Power Without External Memory:
The paper demonstrates that both the Transformer and Neural GPU are Turing complete based on their ability to compute and access internal dense representations of data, without the need for external memory. This is a significant finding because it shows that these models possess a high level of computational power within their bounded architectures​​.

- Arbitrary Precision Assumption and its practical caveat:
A critical aspect of the proof is the assumption of arbitrary precision for internal representations. This is particularly important for storing and manipulating positional encodings in Transformers. However, in practical implementations, neural networks operate with fixed precision due to hardware constraints. Therefore, when fixed precision is used, the Transformer loses its Turing completeness, as the effect of positional encodings becomes equivalent to just increasing the size of the input alphabet​​.

### On the Computational Power of Transformers and its Implications in Sequence Modeling (Bhattamishra et al. 2020)
https://aclanthology.org/2020.conll-1.37.pdf 

- We first provide an alternate and simpler proof to show that vanilla Transformers are Turing-complete and then we prove that Transformers with only positional masking and without any positional encoding are also Turing-complete. We further analyze the necessity of each component for the Turing-completeness of the network; interestingly, we find that a particular type of residual connection is necessary.

"
Theoretical work on Transformers was initiated
by Perez et al. ´ (2019) who formalized the notion of
Transformers and showed that it can simulate a Turing machine given arbitrary precision ... Hahn
(2020) showed some limitations of Transformer
encoders in modeling regular and context-free languages
"

```plaintext
Take-Home Messages. We showed that the order information can be provided either in the form
of explicit encodings or masking without affecting computational power of Transformers. The
decoder-encoder attention block plays a necessary
role in conditioning the computation on the input
sequence while the residual connection around it is
necessary to keep track of previous computations.
The feedforward network in the decoder is the only
component capable of performing computations
based on the input and prior computations. Our
experimental results show that removing components essential for computational power inhibit the
model’s ability to perform certain tasks. At the
same time, the components which do not play a
role in the computational power m
```

```plaintext
Although our proofs rely on arbitrary precision,
which is common practice while studying the
computational power of neural networks in theory
(Siegelmann and Sontag, 1992; Perez et al. ´ , 2019;
Hahn, 2020; Yun et al., 2020), implementations in
practice work over fixed precision settings. However,
our construction provides a starting point to
analyze Transformers under finite precision. Since
RNNs can recognize all regular languages in finite
precision (Korsky and Berwick, 2019), it follows
from our construction that Transformer can also
recognize a large class of regular languages in finite precision.
At the same time, it does not imply
that it can recognize all regular languages given
the limitation due to the precision required to encode positional
information. We leave the study of
Transformers in finite precision for future work.
```

### On the Computational Power of Decoder-Only Transformer Language Models (Roberts, 2023) 
https://arxiv.org/abs/2305.17026

- In this paper we prove the ability of decoder-only transformer models to simulate an arbitrary RNN and are therefore computationally universal.

> very minimal

While the vanilla transformer is known to be Turing Complete (Pérez et al., 2019; Bhattamishraet al., 2020), this does not naturally extend to deocder-only models. Further, no formal evaluation of the computational expressivity exists for the decoder-only transformer architecture. In this paper:
1. We show that the decoder-only transformerarchitecture is Turing complete
2. We show that this result holds even for single layer, single attention head decoder-onlyarchitectures
3. We establish a minimum vector dimensionality, relative to the token embedding size, necessary for Turing completeness
4. We classify transformer models as B machines(Wang, 1957) and identify important future work for en situ computational expressivity
> some clarity:

```plaintext
It is important to point out, seq-to-seq models are not
themselves Turing machines as they do not typically possess
the ability to overwrite a space on their “tape" (the
output vector sequence). Rather, they are much more like the
variant of computational machine studied by (Wang, 1957) called B
machines. B machines are sometimes informally
referred to as non-erasing Turing machines (Neary
et al., 2014). Wang showed that the ability to erase
(or overwrite) is not fundamental to computational
universality. However, he does so by making use of
“auxiliary squares". That is, the machine has free
usage of space to store the results of intermediate
or auxiliary calculations. Wang notes that:
It remains an open question whether we can dispense with auxiliary squares and still be able to
compute all recursive functions by programs consisting of only basic steps.
It remains an important, though apparently unconsidered point. RNNs and decoder-only transformer models are
likewise assumed to be unconstrained regarding the content of their output when
computing recursive functions. However, in many
applications, the output is designed to be constrained to the outputs of some induced function
with a given time delay between samples (like is
the case in natural language). Limiting the outputs
of an RNN in this way violates the assumption
regarding access to Wang’s “auxiliary squares".
A homo-morph of Wang’s question can be
stated in terms of recursion theory, drawing on the
Turing-Church conjecture we may equivalently ask
whether all partial recursive functions are implementable without access to auxiliary computational
space. The answer would seem to be no as this
should limit the network to only calculating primitive recursive functions (those calculable via for
loops) at best. However, this is far from a formal
evaluation.
```

#### *This answers a lot of questions, ir builds on the further improvements made by Bhattamishra et al. The issues about clarity which surrounded the initial paper by Perez have been adressed at this point. Proof rests on showing equivalence with RNN which is proven to be universal*


### Overview of concepts to talk about in tentative order:

- from wikipedia on Universal Turing Machine:
> A universal Turing machine can calculate any recursive function, decide any recursive language, and accept any recursively enumerable language. According to the Church–Turing thesis, the problems solvable by a universal Turing machine are exactly those problems solvable by an algorithm or an effective method of computation, for any reasonable definition of those terms. For these reasons, a universal Turing machine serves as a standard against which to compare computational systems, and a system that can simulate a universal Turing machine is called Turing complete.

- talk about transformer
  - Turing complete theoretically even without positional encoding
  - Not Turing complete empirically, limited precision, is able to recognize certain formal languages, heavily dependent on positional encoding to work.

---

### First concrete plan assembled from the references:

1. **Definition of Turing Completeness**: A computational system is considered Turing complete if it can simulate any Turing machine. This means it can compute anything that is computable, given enough time and memory.

2. **Transformers and Turing Completeness**:
    - The initial paper by Pérez et al. (2019) demonstrated that transformers are theoretically Turing complete. This conclusion is based on the assumption that transformers have arbitrary precision in their internal representations.
    - Turing completeness in this context implies that transformers, in theory, can perform any computation that a Turing machine can, assuming they are not limited by practical constraints like finite precision and finite memory.

3. **Practical Limitations**:
    - In practical implementations, transformers are limited by hardware constraints, most notably fixed precision (like float32). This limitation is significant because it affects the ability of transformers to handle computations requiring very high precision.
    - Furthermore, transformers, particularly those without external memory, rely heavily on their internal structure and precision to handle computations. The absence of external memory means that transformers must use their internal capacity (like layers, heads, and internal states) to store and process information.

4. **Stack Emulation and Turing Completeness**:
    - The ability to emulate a stack is a specific capability often associated with processing nested or hierarchical structures. While it's a useful feature for certain types of computations, it's not a requirement for a system to be Turing complete.
    - Transformers' limitations in emulating stacks, as highlighted in Hahn's paper, point to challenges in handling certain types of structured tasks. However, this does not directly translate to a lack of Turing completeness. It rather indicates that while transformers can theoretically compute anything a Turing machine can, they might not be the most efficient or effective tool for every kind of computation, especially those involving deeply nested or recursive structures.

5. **Theoretical vs. Empirical Turing Completeness**:
    - Theoretically, transformers are Turing complete under the assumption of arbitrary precision. This aligns with the classical approach in theoretical computer science where such idealized assumptions are common.
    - Empirically, in real-world implementations, transformers may not exhibit Turing completeness due to the aforementioned practical constraints. This does not negate their theoretical computational power but highlights the gap between theoretical models and practical applications.

6. **Future Research and Practical Implications**:
    - The current state of research suggests a path forward where the computational power of transformers can be further explored, especially in the context of fixed precision and practical constraints.

So while transformers are theoretically Turing complete, their practical utility and efficiency for specific types of computations, especially those requiring high precision or stack-like processing, are subject to ongoing research and development.

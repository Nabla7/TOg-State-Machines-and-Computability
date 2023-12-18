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
  - > Theoretical study of transformers was initiate by Perez et al. (2019), who theoretically studied the ability of Seq2Seq transformers to emulate the computation of Turing machines.
   - > A related, though different, strand of research has investigated the power of neural networks to model Turing machines. A classical result (Siegelman and Sontag, 1995) states that—given unlimited computation time—recurrent networks can emulate the computation of Turing machines. Very recently, Perez et al. (2019) have shown the same result for both (argmax-attention) Transformers and Neural GPUs. The crucial difference between these studies and studies of language recognition is that, in these studies, the networks are allowed to perform unbounded recurrent computations, arbitrarily longer than the input length.

### Summary again :

- Computational Power Without External Memory:
The paper demonstrates that both the Transformer and Neural GPU are Turing complete based on their ability to compute and access internal dense representations of data, without the need for external memory. This is a significant finding because it shows that these models possess a high level of computational power within their bounded architectures​​.

- Arbitrary Precision Assumption and its practical caveat:
A critical aspect of the proof is the assumption of arbitrary precision for internal representations. This is particularly important for storing and manipulating positional encodings in Transformers. However, in practical implementations, neural networks operate with fixed precision due to hardware constraints. Therefore, when fixed precision is used, the Transformer loses its Turing completeness, as the effect of positional encodings becomes equivalent to just increasing the size of the input alphabet​​.

### Overview of concepts to talk about in tentative order:

- from wikipedia on Universal Turing Machine:
> A universal Turing machine can calculate any recursive function, decide any recursive language, and accept any recursively enumerable language. According to the Church–Turing thesis, the problems solvable by a universal Turing machine are exactly those problems solvable by an algorithm or an effective method of computation, for any reasonable definition of those terms. For these reasons, a universal Turing machine serves as a standard against which to compare computational systems, and a system that can simulate a universal Turing machine is called Turing complete.

- talk about transformer
  - Turing complete theoretically even without positional encoding
  - Not Turing complete empirically, limited precision, is able to recognize certain formal languages, heavily dependent on positional encoding to work.

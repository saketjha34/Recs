# **Comprehensive NLP and Transformer Architecture Concepts**

---

### **1. What is BERT, and how does it differ from GPT?**

**BERT (Bidirectional Encoder Representations from Transformers)** is an **encoder-only** transformer model developed by Google (2018). It processes text bidirectionally, meaning it learns context from both left and right simultaneously.
**GPT (Generative Pretrained Transformer)**, by OpenAI, is a **decoder-only** architecture that generates text **autoregressively** — predicting the next token given the previous ones.

| Feature            | BERT                                                            | GPT                                       |
| ------------------ | --------------------------------------------------------------- | ----------------------------------------- |
| Architecture       | Encoder-only                                                    | Decoder-only                              |
| Training Objective | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) | Autoregressive Next-Token Prediction      |
| Context            | Bidirectional                                                   | Left-to-right                             |
| Use Case           | Text understanding                                              | Text generation                           |
| Example Tasks      | Classification, QA                                              | Summarization, Dialogue, Story Generation |

---

### **2. What is Masked Language Modeling (MLM)?**

MLM is a **self-supervised** learning objective used in BERT.
A percentage of tokens (typically 15%) are replaced by a special token `[MASK]`, and the model predicts the original token based on its **context**.

**Example:**
Input: “The cat [MASK] on the mat.”
Target: “The cat sat on the mat.”

**Mathematical Objective:**
[
\mathcal{L}*{MLM} = - \sum*{i \in M} \log P(w_i | w_{\backslash i})
]
where (M) is the set of masked tokens and (w_{\backslash i}) denotes all unmasked context tokens.

---

### **3. What is Next Sentence Prediction (NSP)?**

NSP helps BERT understand **sentence-level relationships**.
Given two sentences (A) and (B), the model predicts whether (B) is the actual next sentence following (A).

**Loss:**
[
\mathcal{L}_{NSP} = - [y \log p + (1-y) \log (1-p)]
]
where (y = 1) if B follows A, else (y = 0).

---

### **4. What are Prompt Engineering and Few-Shot Learning in LLMs?**

* **Prompt Engineering:** Crafting input prompts strategically to guide LLMs toward desired behavior without retraining.
  Example: “Summarize the following in one sentence: …”

* **Few-Shot Learning:** Providing a few examples in the prompt itself so the model infers task structure.
  Example:

  ```
  Translate English to French:
  Dog → Chien
  Cat → Chat
  Tree → ?
  ```

  Output: “Arbre”

---

### **5. How does GPT generate text autoregressively?**

GPT predicts the next token (w_t) given all previous tokens ((w_1, w_2, …, w_{t-1})) using masked self-attention.

**Formula:**
[
P(w_1, w_2, \dots, w_n) = \prod_{t=1}^{n} P(w_t | w_1, \dots, w_{t-1})
]

This is achieved by masking future tokens to prevent the model from “looking ahead”.

---

### **6. What is Fine-Tuning, and How is it Different from Pretraining?**

| Aspect    | Pretraining                     | Fine-Tuning                        |
| --------- | ------------------------------- | ---------------------------------- |
| Data      | Large, unlabeled corpus         | Small, task-specific labeled data  |
| Objective | Self-supervised (e.g., MLM)     | Supervised (e.g., classification)  |
| Purpose   | Learn general language features | Adapt to specific downstream tasks |

Pretraining gives the model broad linguistic knowledge, while fine-tuning tailors it to a specific task.

---

### **7. Difference Between Encoder-Only, Decoder-Only, and Encoder–Decoder Architectures**

| Type            | Example  | Directionality               | Typical Tasks              |
| --------------- | -------- | ---------------------------- | -------------------------- |
| Encoder-only    | BERT     | Bidirectional                | Text understanding         |
| Decoder-only    | GPT      | Unidirectional               | Text generation            |
| Encoder–Decoder | T5, BART | Bi (Encoder) + Uni (Decoder) | Translation, Summarization |

---

### **8. Advantages of Transformers Over RNNs**

1. **Parallelization:** All tokens are processed simultaneously.
2. **Long-Range Dependencies:** Attention allows direct connection between distant tokens.
3. **Stable Gradients:** No vanishing gradient problem due to skip connections.
4. **Scalability:** Easily trained on large datasets with GPUs.

---

### **9. Role of Positional Encoding in Transformers**

Transformers lack recurrence, so **positional encodings** add order information.

**Formulas:**
[
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
]
[
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
]

Each position gets a unique encoding vector that can be added to word embeddings.

---

### **10. What is Attention in NLP?**

Attention allows a model to **focus on important words** while processing input.
For a query (Q), keys (K), and values (V):

[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

This computes weighted averages of value vectors, where weights represent relevance.

---

### **11. Self-Attention vs Cross-Attention**

| Type            | Definition                                 | Used In                                    |
| --------------- | ------------------------------------------ | ------------------------------------------ |
| Self-Attention  | Q, K, V come from the same sequence        | Encoder or Decoder                         |
| Cross-Attention | Q comes from decoder, K and V from encoder | Encoder–Decoder models (e.g., translation) |

---

### **12. Purpose of Residual Connections and Layer Normalization**

Residual connections stabilize gradients in deep models:

[
x_{out} = x_{in} + \text{Layer}(x_{in})
]

Layer normalization normalizes activations for each token:

[
\text{LN}(x) = \frac{x - \mu}{\sigma}
]

These improve training stability and convergence speed.

---

### **13. How Do LSTMs and GRUs Solve Vanishing Gradient Problems?**

They use **gating mechanisms** to control information flow.

**LSTM equations:**
[
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
]
[
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
]
[
c_t = f_t * c_{t-1} + i_t * \tanh(W_c[h_{t-1}, x_t] + b_c)
]
[
h_t = o_t * \tanh(c_t)
]

This allows gradients to flow through (c_t) over long sequences.

---

### **14. How is Perplexity Used to Evaluate Language Models?**

Perplexity measures how well a model predicts a sample.

[
PPL = e^{-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i})}
]

Lower perplexity = better predictive performance.

---

### **15. Metrics to Evaluate a Language Translation Model**

| Metric    | Description                                    | Notes                    |
| --------- | ---------------------------------------------- | ------------------------ |
| BLEU      | N-gram overlap between candidate and reference | Standard metric          |
| ROUGE     | Recall-based overlap                           | Common for summarization |
| METEOR    | Considers synonyms and stemming                | Linguistically rich      |
| BERTScore | Embedding-based similarity                     | Context-aware            |
| COMET     | Learned metric using embeddings                | SOTA metric              |

---

### **16. Why Are Recurrent Models Slow to Train Compared to Transformers?**

RNNs process tokens **sequentially**, meaning no parallel computation.
Transformers use self-attention that allows all tokens to be processed **in parallel**, making them faster on GPUs/TPUs.

---

### **17. What is Sequence-to-Sequence (Seq2Seq) Architecture?**

Seq2Seq uses an encoder to compress input into a fixed-length vector and a decoder to generate output sequentially.

Example: Neural machine translation.
Encoder → Context Vector → Decoder.

[
h_t^{enc} = f(x_t, h_{t-1}^{enc}), \quad y_t = g(h_{t-1}^{dec}, y_{t-1})
]

---

### **18. What is Bag-of-Words (BoW), and Its Limitations?**

BoW represents text as word-count vectors, ignoring order.

| Strength                  | Limitation                   |
| ------------------------- | ---------------------------- |
| Simple and fast           | Ignores word order           |
| Effective for short texts | Sparse, high-dimensional     |
| Easy to compute           | Fails with synonyms/polysemy |

---

### **19. What is Natural Language Processing (NLP)?**

NLP enables machines to understand, interpret, and generate human language.
It combines **linguistics**, **statistics**, and **machine learning**.

Applications include:

* Sentiment analysis
* Machine translation
* Information retrieval
* Chatbots and Q&A systems

---

### **20. Main Challenges in NLP**

1. Ambiguity (lexical, syntactic, semantic)
2. Context understanding
3. Sarcasm, idioms, and cultural bias
4. Multilinguality and low-resource languages
5. Domain adaptation

---

### **21. What is Tokenization, and Why Is It Important?**

Tokenization splits raw text into manageable units — words, subwords, or characters.
It is crucial for models to handle inputs uniformly.

Example:
“Transformers are powerful” → [“Transform”, “##ers”, “are”, “powerful”]

---

### **22. Intuition Behind Attention Mechanism**

Attention dynamically learns which parts of the input are relevant to each token.
In translation, for example, the word “bank” in “river bank” focuses on “river,” not “money.”

Conceptually:
Input → Attention Weights → Weighted Context Vector → Output

---

### **23. What is Cosine Similarity, and Why Is It Used for Embeddings?**

Cosine similarity measures the angle between two embedding vectors:

[
\cos(\theta) = \frac{A \cdot B}{|A| |B|}
]

It captures semantic similarity independent of magnitude, making it ideal for comparing word embeddings.

---

### **24. Intuition Behind Word2Vec**

Word2Vec learns dense vector representations of words using two architectures:

1. **CBOW:** Predicts the target word given context words.
   Objective: maximize ( P(w_t | context) )
2. **Skip-Gram:** Predicts surrounding words given the target word.
   Objective: maximize ( \sum_{t} \sum_{c \in context(t)} \log P(w_c | w_t) )

Result: Words with similar meanings (e.g., “king”, “queen”, “royal”) are close in vector space.
Semantic relationships emerge algebraically:
[
\text{vec}("king") - \text{vec}("man") + \text{vec}("woman") \approx \text{vec}("queen")
]
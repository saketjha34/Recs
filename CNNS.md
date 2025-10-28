### 🧠 **1. What is the role of filters/kernels in CNNs?**

Filters (or kernels) are small matrices that **slide over the input image** and perform **convolution operations** to detect specific local patterns — like edges, textures, or corners.
Each filter learns to detect a particular feature during training.

* Early layers → detect simple edges or colors
* Deeper layers → detect complex shapes or objects

✅ **Key idea:** Filters extract **spatial hierarchies of features** automatically from raw pixel data.

---

### 🖼️ **2. Why do CNNs work well for images?**

CNNs work well for images because they **exploit spatial locality and parameter sharing**:

* **Local connectivity:** Each neuron looks only at a small region (local receptive field).
* **Weight sharing:** The same filter is applied across the entire image → fewer parameters.
* **Translation invariance:** The same pattern (e.g., edge) can be detected anywhere in the image.
  Together, this makes CNNs **efficient and robust** for vision tasks.

---

### 🧩 **3. What is the purpose of pooling layers?**

Pooling layers **reduce the spatial dimensions** (width × height) of feature maps while retaining important features.
This:

* Decreases computation
* Reduces overfitting
* Provides **translation invariance**

---

### 🔍 **4. Difference between Max Pooling and Average Pooling**

| Type                | What it does                               | Effect                                        |
| ------------------- | ------------------------------------------ | --------------------------------------------- |
| **Max Pooling**     | Takes the **maximum value** in each region | Captures **strongest/most prominent** feature |
| **Average Pooling** | Takes the **average value** of the region  | Retains **overall smooth features**           |

✅ Max pooling is more common since it preserves sharp, discriminative signals.

---

### ⚙️ **5. How do stride and padding affect the output size of a convolution?**

* **Stride:** Number of pixels the filter moves per step.

  * Larger stride → smaller output → faster but less detailed.
  * Stride = 1 preserves detail.
* **Padding:** Adding zeros around the border to control spatial size.

  * **Valid (no padding):** output shrinks.
  * **Same (with padding):** output size same as input.

Formula:
[
\text{Output size} = \frac{(N + 2P - F)}{S} + 1
]
where
N = input size, F = filter size, P = padding, S = stride.

---

### 🧮 **6. What are 1×1 convolutions used for?**

1×1 convolutions are used to:

* **Reduce or expand channel dimensions** (bottleneck layers in ResNet/Inception).
* **Add non-linearity** without changing spatial dimensions.
* Combine features across channels efficiently.

✅ Think of it as a **feature mixer** across channels.

---

### 🔊 **7. How would you modify a CNN to work on non-image data (audio or time series)?**

* For **audio/speech:** use 1D convolutions over time (instead of 2D over width/height).
* For **time series:** input is (sequence_length × features), and apply 1D convolutions.
* For **text:** represent words as embeddings, then convolve over sequences.

✅ The key idea: choose kernel size based on **temporal neighborhood** rather than spatial.

---

### 🔗 **8. What is a skip connection (in ResNet)?**

A skip (residual) connection **adds the input of a layer directly to its output**, i.e.
[
y = F(x) + x
]
It helps the network **learn residual mappings** instead of direct transformations.
Benefits:

* Solves **vanishing gradient** problem.
* Makes training of **very deep networks** (100+ layers) possible.
* Allows gradients to **flow easily** through identity paths.

---

### ⚡ **9. What is the "dying ReLU" problem?**

If a neuron’s input becomes negative, ReLU outputs zero, and the gradient becomes zero.
Once this happens, that neuron may **never activate again** — effectively "dead".
This happens due to **large learning rates** or **imbalanced data**.

✅ Fixes:

* Use **Leaky ReLU**, **ELU**, or **parametric ReLU (PReLU)**
* Use smaller learning rates or better initialization.

---

### 🔥 **10. Why might you prefer ReLU over more complex functions in CNNs?**

* **Simple & fast:** ReLU = `max(0, x)` → easy to compute.
* **Non-saturating gradient:** avoids vanishing gradient issues.
* **Sparse activation:** many zeros → efficient computation and regularization.
* Works well empirically across almost all CNN architectures.

✅ Though newer activations (Swish, GELU) can yield small gains, ReLU’s **speed and stability** make it the default choice.

---

### 🧠 **11. Compare ReLU, Sigmoid, Tanh, GELU, and Swish**

| Activation  | Formula / Idea            | Pros                                    | Cons                                          | Common Use                |
| ----------- | ------------------------- | --------------------------------------- | --------------------------------------------- | ------------------------- |
| **ReLU**    | max(0, x)                 | Fast, sparse, avoids vanishing gradient | Dying ReLU                                    | Default in CNNs           |
| **Sigmoid** | 1 / (1 + e^-x)            | Smooth, bounded                         | Vanishing gradient, outputs not zero-centered | Legacy, output layers     |
| **Tanh**    | (e^x - e^-x)/(e^x + e^-x) | Zero-centered, smooth                   | Still vanishes for large                      | RNNs                      |
| **Swish**   | x·sigmoid(x)              | Smooth, non-monotonic, better flow      | Slightly slower                               | EfficientNet, newer CNNs  |
| **GELU**    | x·Φ(x) (Gaussian CDF)     | Smooth, probabilistic gating            | More compute                                  | Transformers, modern CNNs |

✅ **Takeaway:** ReLU = simplicity & stability; Swish/GELU = performance gains in deeper models.

---

### 💧 **12. What is dropout, and why is it used?**

Dropout randomly **“drops” (sets to zero)** some neurons during training with probability *p*.
Purpose:

* Prevents **co-adaptation** of neurons.
* Forces the network to **learn redundant, generalizable features**.
* Acts as **regularization** to reduce overfitting.

---

### 🧩 **13. How does dropout prevent overfitting?**

It makes the network behave like an **ensemble of many smaller networks** that share weights.
Because each forward pass activates a random subset of neurons, the model:

* Can’t rely on specific neurons.
* Learns robust feature representations.
* Generalizes better to unseen data.

---

### 🎯 **14. How would you choose a good dropout rate?**

Typical values depend on layer type:

* **Input layers:** 0.1 – 0.3
* **Hidden layers:** 0.4 – 0.6
* **Convolutional layers:** smaller (0.1 – 0.3) or use **SpatialDropout**
  Choose by **tuning** on validation data — too high → underfitting; too low → overfitting.
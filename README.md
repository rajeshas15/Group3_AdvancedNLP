# 📘 Group3_AdvancedNLP – VAE for Sentence Generation

**Task:**  
Re-implement the main NLP algorithm used in the referenced research paper using **Python** and **PyTorch**. The implementation focuses on a Variational Autoencoder (VAE) applied to sentences. Deliverables include: source code, input data, and modular components organized by responsibility.

---

## 📂 Project Structure & Responsibilities

### 1. 🧹 Input Data Preprocessing *(Himanshu)*
- Tokenization (e.g., Penn Treebank token rules)
- Vocabulary construction & indexing
- Padding / batching for LSTM training

---

### 2. 🧠 Core VAE Architecture for Sentences *(Himanshu)*
- Single-layer LSTM Encoder
- Single-layer LSTM Decoder
- Latent space with Gaussian prior
- Softplus activation to ensure positive variance
- Reparameterization trick for sampling \( z \sim \mathcal{N}(\mu, \sigma^2) \)
- Highway network (4 layers) to parametrize \( \mu \) and \( \log \sigma^2 \)
- Decoder initialization from latent vector \( z \)
- Optional: Right-to-left decoding variant

---

### 3. 🏋️ Training Procedure *(Himanshu)*
- Stochastic Gradient Descent (SGD)
- Alternating encoder-decoder training (non-joint strategy)
- KL Annealing (gradual KL term increase)
- Word Dropout (masking previous token inputs)
- History-less decoding (no autoregressive signal)

---

### 4. ⚖️ Loss Function *(Rajesha)*
- Reconstruction loss (cross-entropy)
- KL divergence loss between posterior \( q(z|x) \) and prior \( p(z) \)
- ELBO Objective: `Reconstruction Loss + KL Divergence`

---

### 5. 📈 Evaluation: Language Modeling *(Rajesha)*
- Negative Log Likelihood (NLL)
- Perplexity (PPL)

---

### 6. 🧭 Improved Inference *(Rale)*
- Iterated Conditional Modes (ICM) decoding
- 3-step beam search (beam size = 5)

---

### 7. 🧪 Adversarial Evaluation *(Rale)*
- Dataset with 50% human-written vs. 50% generated completions
- Bag-of-unigrams logistic regression classifier
- LSTM-based classifier
- Early stopping (80/10/10 train/val/test split)
- Accuracy reporting for classifier performance

---

### 8. 🌌 Latent Space Analysis *(Rasheed)*
- Sampling from prior and posterior distributions
- Study of word dropout effects on latent structure
- Homotopies: interpolate between latent vectors and decode sequences

---

### 9. 🧾 Text Classification Task *(Team)*
- Use latent vectors as classification features
- Train classifier (e.g., sentiment, topic detection)
- Implementation depends on dataset availability

---

### 10. 🎯 Hyperparameter Tuning *(Rasheed)*
- Bayesian optimization of:
  - Learning rate
  - Dropout
  - Latent dimension size (z-dim)

---

## 🔧 Technologies Used
- Python 3.11
- PyTorch
- NumPy, SciPy
- Matplotlib / Seaborn (for plots)
- Scikit-learn
- [Optional] TensorBoard for training visualization

---

## 📦 Running the Code
> Detailed instructions to set up the environment, install dependencies, and run training will be provided in a separate `INSTALL.md` or in the main `README.md` below this section.

---

## 📚 Reference
Inspired by the paper:  
📄 *Generating Sentences from a Continuous Space*  
Bowman et al., 2016 – [Link to Paper](https://arxiv.org/abs/1511.06349)

Task: Re-Implement the main NLP algorithm used in this paper yourself using Python and PyTorch (as far as you get with it) (submit the Python code and input data you used)
1.	Input Data Preprocessing (Himanshu)
    •	Tokenization (e.g., Penn Treebank token rules)
    •	Vocabulary construction & indexing 
    •	Padding / batching for LSTM training
2.	Core VAE Architecture for Sentences (Himanshu)
    •	Single-layer LSTM Encoder
    •	Single-layer LSTM decoder
    •	Latent space with Gaussian prior
    •	Softplus activation to ensure positive variance
    •	Reparameterization trick for sampling z∼N(μ,σ2)
    •	Highway network (4 layers) to parametrize μ and log σ2
    •	Decoder initialization from latent vector z (linear layer to decoder hidden state)
    •	Right-to-left decoding (optional, mentioned as a variant tried)
3.	Training Procedure (Himanshu)
    •	Stochastic Gradient Descent (SGD)
    •	Alternating encoder-decoder training instead of simultaneously (non-joint training, as described)
    •	KL Annealing – gradual increase of KL term weight in loss
    •	Word Dropout (masking previous token inputs during training)
    •	History-less decoding (remove autoregressive signal during training input)
4.	Loss Function (Rajesha)
    •	Reconstruction loss (cross-entropy between decoded tokens and ground truth)
    •	KL divergence loss between posterior q(z∣x) and prior p(z)
    •	ELBO objective = Reconstruction loss + KL divergence
5.	Evaluation: Language Modeling (Rajesha)
    •	Compute Negative Log Likelihood (NLL)
    •	Compute Perplexity (PPL)
6.	Improved Inference (Rale)
    •	Iterated conditional modes (ICM) decoding strategy
    •	3-step search with beam size = 5
7.	Adversarial Evaluation (Rale)
    •	Create dataset with 50% human-written, 50% generated completions
    •	Train Bag-of-unigrams logistic regression classifier
    •	Train LSTM-based classifier
    •	Early stopping with 80/10/10 train/val/test split
    •	Report accuracy to measure classifier’s ability to detect generated content
8.	Latent Space Analysis (Rasheed)
    •	Sampling from prior and posterior
    •	Effect of word dropout on latent space structure
    •	Latent interpolations (homotopies): interpolate between two latent vectors and decode at each step
9.	Text Classification Task
    •	Use VAE-encoded latent representations as features
    •	Train classifier for text classification task (e.g., sentiment or topic)
    (Implementation depends on the available dataset; might need adaptation)
10.	Hyperparameter Tuning (Rasheed)
    •	Bayesian optimization for hyperparameters (e.g., learning rate, dropout, z-dim)

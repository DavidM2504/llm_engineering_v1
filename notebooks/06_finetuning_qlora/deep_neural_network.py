# =============================================================================
# Program: deep_neural_network.py
#
# PURPOSE:
#   Defines and trains a deep residual neural network that predicts product
#   prices from text summaries. The model takes a bag-of-words representation
#   of an item's summary (produced by batch.py or preprocessor.py) and
#   outputs a predicted price in USD.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - Input data comes from Item objects defined in items.py. Each item must
#     have a populated .summary field (set by batch.py or preprocessor.py)
#     and a .price field.
#   - The trained model's inference() method is called by evaluator.py when
#     running price predictions during evaluation.
#   - loaders.py and items.py provide the train/val Item lists passed to
#     DeepNeuralNetworkRunner.__init__().
# =============================================================================

import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer


# =============================================================================
# Model Architecture
# =============================================================================

class ResidualBlock(nn.Module):
    """
    A single residual (skip-connection) block used as a building unit inside
    DeepNeuralNetwork. The block applies two linear transformations with
    LayerNorm + ReLU + Dropout, then adds the original input back (skip
    connection). This helps gradients flow through many layers and prevents
    vanishing gradients in the deep network.

    Structure of one block:
        x --> Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> LayerNorm
          |                                                             |
          +-----(skip connection: add original x)----------------------+
          --> ReLU --> output
    """

    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        # Two-layer transform; output size matches input size for the skip connection
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x           # Save input for the skip connection
        out = self.block(x)    # Transform
        out += residual        # Add skip connection (residual learning)
        return self.relu(out)  # Final activation


class DeepNeuralNetwork(nn.Module):
    """
    Full price-prediction network composed of:
      1. An input projection layer (sparse 5000-dim BoW → hidden_size)
      2. A stack of residual blocks (default: 8 blocks for a 10-layer total)
      3. A single output neuron producing the normalized log-price

    Default configuration: 10 layers, 4096 hidden units, 0.2 dropout.
    With these defaults the network has several tens of millions of parameters.
    """

    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # --- Input layer: project sparse BoW vector into dense hidden space ---
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # --- Stack of residual blocks (num_layers - 2 because input & output
        #     layers each count as one layer) ---
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # --- Output layer: project to a single scalar (predicted price) ---
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        # Pass through each residual block sequentially
        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)  # Returns normalized log-price


# =============================================================================
# Training & Inference Runner
# =============================================================================

class DeepNeuralNetworkRunner:
    """
    High-level wrapper that handles the complete ML pipeline:
      - Vectorizing item summaries (text → sparse BoW via HashingVectorizer)
      - Log-normalizing prices for stable training
      - Setting up optimizer, scheduler, and data loaders
      - Running the training loop with validation after each epoch
      - Saving/loading model weights
      - Running inference on a single Item at prediction time

    Usage pattern:
        runner = DeepNeuralNetworkRunner(train_items, val_items)
        runner.setup()
        runner.train(epochs=10)
        runner.save("model.pt")
        price = runner.inference(some_item)   # Called from evaluator.py
    """

    def __init__(self, train, val):
        """
        train : list of Item objects (with .summary and .price) for training
        val   : list of Item objects for validation
        Both lists come from items.py / loaders.py after batch.py has populated
        each item's .summary field.
        """
        self.train_data = train
        self.val_data = val

        # These are all populated by setup()
        self.vectorizer = None
        self.model = None
        self.device = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.train_loader = None
        self.y_mean = None   # Mean of log-prices (used to un-normalize predictions)
        self.y_std = None    # Std  of log-prices (used to un-normalize predictions)

        # Fix random seeds for reproducibility across numpy, PyTorch, and CUDA
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        """
        Prepare all components before training:
          1. Vectorize item summaries using a HashingVectorizer (5000 features,
             binary term presence). HashingVectorizer is stateless — no fitting
             needed — and memory-efficient for large vocabularies.
          2. Convert prices to log-space and normalize to zero mean / unit std.
             This stabilises training because raw prices span several orders of
             magnitude ($0.50 to $999).
          3. Instantiate DeepNeuralNetwork, move to the best available device
             (CUDA > MPS > CPU).
          4. Set up AdamW optimizer, L1 loss, cosine LR scheduler, and
             DataLoader with shuffled mini-batches.
        """
        # --- Text vectorization ---
        # HashingVectorizer converts item.summary text (from batch.py /
        # preprocessor.py) into a 5000-dimensional binary bag-of-words vector
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)

        train_documents = [item.summary for item in self.train_data]
        X_train_np = self.vectorizer.fit_transform(train_documents)
        self.X_train = torch.FloatTensor(X_train_np.toarray())

        # Extract prices from Item objects (defined in items.py)
        y_train_np = np.array([float(item.price) for item in self.train_data])
        self.y_train = torch.FloatTensor(y_train_np).unsqueeze(1)

        val_documents = [item.summary for item in self.val_data]
        X_val_np = self.vectorizer.transform(val_documents)
        self.X_val = torch.FloatTensor(X_val_np.toarray())
        y_val_np = np.array([float(item.price) for item in self.val_data])
        self.y_val = torch.FloatTensor(y_val_np).unsqueeze(1)

        # --- Price normalization: log(price+1), then z-score ---
        # log(price+1) compresses the long tail of expensive items;
        # z-scoring makes the target distribution roughly N(0,1) for the model
        y_train_log = torch.log(self.y_train + 1)
        y_val_log = torch.log(self.y_val + 1)
        self.y_mean = y_train_log.mean()
        self.y_std = y_train_log.std()
        self.y_train_norm = (y_train_log - self.y_mean) / self.y_std
        self.y_val_norm = (y_val_log - self.y_mean) / self.y_std

        # --- Model instantiation ---
        self.model = DeepNeuralNetwork(self.X_train.shape[1])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Deep Neural Network created with {total_params:,} parameters")

        # --- Device selection: prefer GPU (CUDA or Apple MPS), fall back to CPU ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}")

        self.model.to(self.device)

        # L1 loss (mean absolute error in normalised space) is robust to outliers
        self.loss_function = nn.L1Loss()

        # AdamW with weight decay helps regularise the large network
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)

        # Cosine annealing gradually reduces the learning rate each epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

        # DataLoader shuffles mini-batches each epoch to improve generalisation
        self.train_dataset = TensorDataset(self.X_train, self.y_train_norm)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def train(self, epochs=5):
        """
        Run the training loop for the given number of epochs.
        Each epoch:
          - Iterates over all mini-batches, computes L1 loss, back-propagates,
            and clips gradients to prevent explosions in the deep network.
          - Evaluates on the full validation set (no gradient computation).
          - Converts normalised predictions back to raw dollar values to report
            a human-readable mean absolute error (MAE).
          - Steps the cosine LR scheduler.
        """
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in tqdm(self.train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()

                # Gradient clipping: prevents exploding gradients in deep networks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_losses.append(loss.item())

            # --- Validation pass ---
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val.to(self.device))
                val_loss = self.loss_function(val_outputs, self.y_val_norm.to(self.device))

                # Invert normalisation: un-z-score, then exp(x)-1 to get dollars
                val_outputs_orig = torch.exp(val_outputs * self.y_std + self.y_mean) - 1
                mae = torch.abs(val_outputs_orig - self.y_val.to(self.device)).mean()

            avg_train_loss = np.mean(train_losses)
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Val mean absolute error: ${mae.item():.2f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path):
        """Save trained model weights to disk (PyTorch state dict format)."""
        torch.save(self.model.state_dict(), path)

    def load(self, path, device="mps"):
        """
        Load previously saved model weights from disk.
        map_location ensures weights load correctly even if the save/load
        devices differ (e.g. saved on CUDA, loaded on MPS).
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(self.device)

    # -------------------------------------------------------------------------
    # Inference (called by evaluator.py)
    # -------------------------------------------------------------------------

    def inference(self, item):
        """
        Predict the price for a single Item object.
        Steps:
          1. Vectorize item.summary using the same HashingVectorizer fitted
             during setup() — this is the text produced by batch.py or
             preprocessor.py and stored on the Item (see items.py).
          2. Run the model forward pass (no gradient tracking).
          3. Invert the log-normalisation to return a raw dollar amount.
          4. Clamp to >= 0 (prices can't be negative).

        This method is the bridge between this model and evaluator.py, which
        calls it via a predictor function during the Tester.run_datapoint() loop.
        """
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            # Invert normalisation: un-z-score, then reverse log(x+1)
            result = torch.exp(pred * self.y_std + self.y_mean) - 1
            result = result.item()
        return max(0, result)  # Negative prices are not physically meaningful

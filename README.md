Mini-GPT Transformer (French TinyStories)
=========================================

Mini-GPT is a lightweight GPT-style Transformer language model trained from scratch in **PyTorch** on the **French TinyStories** dataset.\
It includes a complete and modular training pipeline with configurable hyperparameters, environment variables, and text generation.

* * * * *

📘 Overview
-----------

This project aims to provide a compact, educational implementation of a **French autoregressive Transformer** model inspired by GPT architectures.

Key features:

-   Implemented fully in **PyTorch**

-   **Causal attention mask** for proper autoregressive behavior

-   Training with **AdamW** optimizer and **OneCycleLR** scheduler

-   Configurable via `config.json` and `.env`

-   Automatic checkpoint saving and resume support

-   Compatible with **Google Colab** and local GPUs

-   Text generation from any custom prompt

* * * * *

⚙️ Project Structure
--------------------

```mini-gpt/
│
├── train.py                 # Training script
├── generate.py              # Text generation script
├── config.json              # Hyperparameter configuration
├── .env                     # Environment variables (dataset, paths)
│
├── model/
│   └── model.py             # MiniGPT model (Transformer architecture)
│
├── dataset/
│   └── text_dataset.py      # Dataset wrapper and tokenization logic
│
└── checkpoints/             # Saved model weights
```

* * * * *

🔧 Installation
---------------

You can run this project either **locally** or in **Google Colab**.

### 1\. Clone the repository

`git clone https://github.com/eric-houzelle/mini-gpt.git
cd mini-gpt`

### 2\. (Optional) Create a virtual environment

```python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3\. Install dependencies

`pip install -r requirements.txt`

* * * * *

⚙️ Configuration
----------------

### `config.json`

Defines model and training hyperparameters.\
Example:

```
{
  "training": {
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "scheduler_max_lr": 0.0003
  },
  "model": {
    "embed_dim": 256,
    "depth": 8,
    "heads": 8,
    "block_size": 128
  },
  "data": {
    "max_texts": 10000,
    "train_split_ratio": 0.9
  }
}
```

### `.env`

Specifies runtime environment variables.

```
DATASET_NAME=iproskurina/TinyStories-French
TOKENIZER_NAME=camembert-base
MODEL_SAVE_PATH=checkpoints/best_miniGPT.pt
```

* * * * *

🧠 Training the Model
---------------------

To train MiniGPT from scratch:

`python train.py`

During training:

-   The model and optimizer states are saved automatically to `checkpoints/best_miniGPT.pt` whenever validation loss improves.

-   Training progress and validation loss are printed to the console.

-   At the end of each epoch, a short text sample is generated to evaluate progress.

* * * * *

✍️ Generating Text
------------------

Once training is complete (or if you already have a checkpoint), run:

```
python generate.py --prompt "Il était une fois un petit robot curieux" --tokens 100
```

This will:

-   Load the model from `MODEL_SAVE_PATH`

-   Tokenize your prompt

-   Generate the specified number of tokens

-   Print the decoded French text to the console

* * * * *

💻 Running on Google Colab
--------------------------

You can also run everything directly in Google Colab.\
Example Colab snippet:

```
!git clone https://github.com/eric-houzelle/mini-gpt.git
%cd mini-gpt
!pip install torch torchvision torchaudio transformers datasets python-dotenv
!python train.py
```

Or to generate text after training:

```
!python generate.py --prompt "Il était une fois" --tokens 120
```

* * * * *

📄 License
----------

This project is released under the MIT License.\
You are free to use, modify, and distribute it for educational or research purposes.
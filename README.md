# 📺 Seinfeld Script Generator — LSTM Text Generation with PyTorch

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/Udacity-Deep_Learning_Nanodegree-02B3E4)

## 🚀 Overview

A character-level LSTM network that generates new Seinfeld TV scripts after training on 9 seasons of real scripts. The model learns dialogue patterns, character voices, and scene structure, then generates new scenes with recognizable Seinfeld-style exchanges between Jerry, George, Elaine, and Kramer. Trained for 20 epochs with a 2-layer LSTM, 300-dim embeddings, and 400 hidden units.

## 📝 Sample Generated Script

```
jerry: (elaine enters)

jerry: (looking for the paper) oh, i got a good one, you know,
and you know, i don't want any trouble.

jerry: you can't take a look for the rest of the life...

jerry: (looking at jerry) hey, hey, i need to get a little more
more than you know what i do.

kramer: yeah!

george: oh, i don't think i would be so funny.

elaine: i can't get the money back!

kramer: well, i got the whole thing on it.

jerry: i think that's it.
```

## ✨ Key Features

- **Word-Level LSTM** — Embedding layer → 2-layer LSTM → fully-connected output over the full vocabulary, predicting the next word given a sequence of 10 previous words
- **Custom Preprocessing** — Word-to-integer lookup tables, punctuation tokenization (replacing `.`, `,`, `!`, `?`, `;`, etc. with unique tokens), and sequence batching with sliding window
- **Hyperparameter Exploration** — Multiple configurations tested (commented in notebook): hidden_dim [256, 400, 450], embedding_dim [300, 350], epochs [4, 10, 13, 20], showing iterative tuning
- **Script Generation** — Seeded with a prime word, generates sequences by sampling from the model's output distribution word by word

## 🧠 Technical Highlights

**Model Architecture:**
```
Input (batch, seq_length=10)
  → Embedding(vocab_size, 300)
  → LSTM(300, 400, n_layers=2, dropout=0.5, batch_first=True)
  → Dropout(0.5)
  → Linear(400, vocab_size)
  → Output: last word scores → softmax → next word
```

- **Training Progression** — Loss decreased from 4.61 (epoch 1) to ~3.5 (epoch 20), showing consistent learning without overfitting
- **Sequence Batching** — Sliding window creates (feature, target) pairs where features are 10 consecutive words and target is the 11th word. Batched into groups of 256 with shuffling
- **Hidden State Management** — LSTM hidden state `(h, c)` initialized to zeros for each batch, with GPU/CPU handling via `train_on_gpu` flag
- **Contiguous View Reshaping** — LSTM output reshaped via `.contiguous().view(-1, hidden_dim)` before the FC layer, then reshaped back to `(batch_size, seq_length, vocab_size)` to extract the last timestep's predictions

## 📊 Hyperparameters

| Parameter | Value |
|---|---|
| Sequence Length | 10 words |
| Batch Size | 256 |
| Embedding Dim | 300 |
| Hidden Dim | 400 |
| LSTM Layers | 2 |
| Dropout | 0.5 |
| Learning Rate | 0.001 |
| Epochs | 20 |
| Optimizer | Adam |

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Model | 2-layer LSTM with word embeddings |
| Dataset | Seinfeld Scripts (9 seasons, Kaggle) |
| NLP | Custom tokenizer + word-to-int vocabulary |
| Environment | Jupyter Notebook, GPU recommended |

## 📁 Project Structure

```
Seinfield-Script-Generator/
├── dlnd_tv_script_generation.ipynb  # Main notebook
├── data/Seinfeld_Scripts.txt        # Training data (9 seasons)
├── generated_script_1.txt           # Sample generated output
├── helper.py                        # Data loading + model save/load
├── problem_unittests.py             # Unit tests for model components
└── workspace_utils.py               # Long-running notebook utility
```

## ⚡ Getting Started

```bash
git clone https://github.com/jashjain21/Seinfield-Script-Generator.git
cd Seinfield-Script-Generator

pip install torch numpy

jupyter notebook dlnd_tv_script_generation.ipynb
```

## 🔍 What This Project Demonstrates

- **Sequence Modeling** — Using LSTMs to learn temporal dependencies in text and generate coherent multi-turn dialogue
- **NLP Preprocessing** — Building a vocabulary, tokenizing punctuation, creating word embeddings, and batching variable-length sequences
- **Text Generation** — Autoregressive generation: feeding each predicted word back as input to produce arbitrarily long scripts
- **Hyperparameter Tuning** — Iterative experimentation with hidden dimensions, embedding sizes, and epoch counts (5 configurations tested)

## 🚧 Limitations / Future Improvements

- **Repetitive Output** — The model tends to repeat phrases like "you know" and "i don't" — temperature-based sampling or top-k/top-p sampling would increase diversity
- **No Character Conditioning** — The model doesn't explicitly know which character is speaking; adding character embeddings could make each voice more distinct
- **Word-Level Only** — A character-level or subword (BPE) model could handle rare words and generate more creative vocabulary
- **No Attention Mechanism** — Adding attention or using a Transformer architecture would improve long-range coherence across dialogue turns
- **No Beam Search** — Generation uses greedy/random sampling; beam search would produce more coherent sequences

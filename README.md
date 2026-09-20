# SebAI V1

**SebAI is an independent language model built from scratch.**

SebAI is an experimental language model created to understand how language models work by building one from the ground up rather than starting with a pretrained model.

The project is written in **Python** and trained with **PyTorch** on Google Colab.

>  SebAI is an experimental research/learning project. It is not intended to compete with modern large language models.

---

## What is SebAI?

SebAI learns language by predicting what comes next.

The model starts with **randomly initialized weights** and gradually learns patterns from its training data through optimization.

The basic idea is:

```text
Training text
     ↓
Tokenization
     ↓
Model
     ↓
Predict next token
     ↓
Calculate loss
     ↓
Backpropagation
     ↓
Update weights
     ↓
Repeat
```

The goal is to build the entire learning pipeline myself and understand each part rather than simply loading an existing pretrained model.

---

## Current Status

 **Active development**

SebAI is currently an experimental model and is still being trained and improved.

---

## Architecture

SebAI is designed to learn language through next-token prediction.

At a high level:

```text
Input
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embedding
  ↓
Language Model
  ↓
Logits
  ↓
Next-token prediction
```

During training, the predicted tokens are compared with the actual next tokens in the dataset.

The resulting loss is used to update the model's weights through backpropagation.

---

## Training

SebAI is trained with **PyTorch**.

Training is currently performed using **Google Colab**, including NVIDIA GPU acceleration when available.

Example training loop:

```python
for batch in dataloader:
    optimizer.zero_grad()

    output = model(batch)

    loss = loss_function(
        output,
        targets
    )

    loss.backward()
    optimizer.step()
```

The model is trained repeatedly over the dataset until it learns increasingly useful statistical patterns.

---

## Dataset

SebAI uses text data for language-model training.

Dataset preparation is an important part of the project because low-quality or duplicated data can negatively affect training.

The long-term goal is to build a **clean, curated dataset** rather than simply throwing huge amounts of scraped text into the model.

---

## Tokenization

SebAI uses tokenization to convert text into numerical representations that the model can process.

For example:

```text
"Hello world"
       ↓
[ token_1, token_2 ]
       ↓
[ 1042, 8921 ]
```

The exact tokenizer and vocabulary are still being developed.

---

## Tech Stack

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| Python                  | Main programming language |
| PyTorch                 | Model training            |
| Google Colab            | Training environment      |
| Hugging Face Datasets   | Dataset processing        |
| Hugging Face Tokenizers | Tokenization              |
| Git / GitHub            | Version control           |

---

## Project Goals

SebAI is primarily a **learning and research project**.

The goal is to understand the technology behind modern language models by implementing the important components myself.

Long-term goals include:

* Building a capable language model from scratch
* Understanding transformer-based architectures
* Developing a custom tokenizer
* Creating better training datasets
* Experimenting with model architectures
* Implementing efficient inference
* Adding RAG
* Building a local AI assistant around SebAI

---

## Why "from scratch"?

SebAI is not based on a pretrained GPT, Llama, Mistral, Qwen, or other language model.

The model begins with randomly initialized parameters and learns from the training data.

This makes the project significantly more difficult, but it also makes it useful for understanding what actually happens during language-model training.

---

## Repository Structure

```text
SebAI/
│
├── data/
│   └── ...
│
├── model/
│   └── ...
│
├── tokenizer/
│   └── ...
│
├── training/
│   └── ...
│
├── inference/
│   └── ...
│
├── checkpoints/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── requirements.txt
└── README.md
```

The structure may change as SebAI develops.

---

## Example

After training, SebAI can generate text by repeatedly predicting the next token:

```text
Input:
The sky is

SebAI:
The sky is blue...
```

Early generations may be incomplete, repetitive, or nonsensical.

That's expected.

The model is learning language from its training data, and improving its ability to generate useful text requires better data, training, architecture, and scale.

---

## Training Progress

Training metrics and experiments will be documented as the project develops.

Example:

```text
Training step: 79,800

Training loss:   ~0.986
Validation loss: ~1.166
```

These numbers are experiment-specific and should not be interpreted as a direct comparison with other language models.

---

## Limitations

SebAI is currently limited by:

* Small training scale
* Limited compute
* Limited dataset size
* Limited model size
* Experimental tokenizer
* Limited training time
* No production-grade inference system

Because of these limitations, SebAI should be considered an **experimental model**, not a production AI assistant.

---

## Philosophy

> **Don't just use AI. Build one.**

SebAI exists to explore how language models actually work by building the system step by step.

The project prioritizes **understanding, experimentation, and ownership** over simply obtaining the best possible benchmark score.

---

## License

This project is open source.

See the `LICENSE` file for details.

---

## Author

**AK74**

Frontend and Backend Developer • Website, Software & AI Engineer

GitHub: [@ssh-ak74](https://github.com/ssh-ak74)

---

⭐ If you're interested in learning how language models work from the ground up, feel free to explore the code.

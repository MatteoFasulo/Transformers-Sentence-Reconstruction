# 📚 Sentence Reconstruction using Transformer Model

## 🚀 Introduction

This project is a comprehensive exploration of the application of deep learning techniques to the problem of sentence reconstruction. The task is to reconstruct the original sequence of a given English sentence from a random permutation of its words. This is a challenging problem due to the inherent complexity of language and the numerous possible permutations of a sentence.

## 🌟 Project Overview

The project is constrained by several factors:

* ❌ **No use of pretrained models:** The model must learn the task from scratch, without the benefit of prior knowledge encoded in models like BERT or GPT.
* 📏 **Model size limitation:** The neural network model size is limited to less than 20M parameters, ensuring efficiency and the ability to train and deploy on modest hardware.
* 🚫 **No postprocessing techniques:** The model’s output must be used directly, without additional processing to select the most likely sequence.
* 📊 **Limited training data:** The model must learn from the provided dataset alone.

## 🛠️ Libraries and Frameworks

The project is implemented in Python, using:

- **Keras** for building and training the neural network model.
- **TensorFlow** as the backend for Keras, providing efficient, GPU-accelerated computations.
- **Hugging Face** library for providing the datasets for training and testing the model.

## ⚙️ Preprocessing and Model Design

The project includes extensive preprocessing of the dataset. This involves tokenizing the sentences, encoding the words as integers, and creating the random permutations. A custom text vectorization layer is created for this purpose.

The model itself is a Transformer, a type of neural network that uses self-attention mechanisms to capture the dependencies between words in a sentence. The model has an encoder-decoder architecture, where the encoder processes the input sentence and the decoder generates the reconstructed sentence.

## 🏗️ Model Architecture

The Transformer model architecture used in this project includes:

- **Embedding Layer**: Converts input tokens into dense vectors of fixed size.
- **Encoder**: Multiple layers of self-attention and feed-forward neural networks.
- **Decoder**: Similar to the encoder but designed to produce the output sequence.
- **Positional Encoding**: Adds information about the position of each token in the sequence.

## 🏋️ Training

The training process involves the following steps:

1. **Data Preparation**: Tokenize and encode the input sentences and their shuffled versions.
2. **Model Initialization**: Define the Transformer model with specified hyperparameters.
3. **Loss Function and Optimizer**: Use cross-entropy loss and Adam optimizer.
4. **Training Loop**: Train the model over several epochs, adjusting weights to minimize the loss.

## 📊 Evaluation

The model is evaluated using a specific metric that measures the accuracy of the reconstructed sentences. This metric finds the longest common subsequence between the original and reconstructed sentences and calculates the ratio of this length to the length of the original sentence. A higher ratio indicates a better reconstruction. The model is tested on a separate test set to assess its performance. The evaluation process involves:

- Generating reconstructed sentences from the shuffled input.
- Comparing the reconstructed sentences with the original sentences.

## 🏆 Results

The proposed Transformer model demonstrated good performance in reconstructing sentences. It outperformed Seq2Seq models with LSTM encoders and decoders in capturing long-term dependencies and syntactic structures.

### 📈 Performance Metrics and Testing

The Transformer model is tested on a set of 3,000 randomly selected instances. The results are promising, with the model achieving an average score of approximately 0.51 with a standard deviation of 0.28. This indicates that the model is able to reconstruct the original sentence with reasonable accuracy.

The model’s architecture, which has less than 10 million parameters, is discussed in detail. The impact of various hyperparameters, such as the number of layers and the embedding dimensions, on the model’s size and performance is analyzed.

## 🔍 Conclusions and Future Work

The Transformer model shows promise in the task of sentence reconstruction. It is able to capture long-term dependencies and syntactic structures in the sentences, outperforming previous LSTM-based Seq2Seq models. This is achieved within the constraints of the project’s parameter limit.

### 💡 Potential Improvements

- **Model Optimization**: Increasing the number of layers and the embedding dimensions to further enhance performance.
- **Custom Loss Function**: Implementing a custom loss function that penalizes shorter sequences could help improve the model's accuracy.
- **Hyperparameter Tuning**: Exploring different hyperparameter configurations to achieve better results.
- **Attention Visualization**: Visualizing attention weights to better understand how the model processes and reconstructs sentences.

The project concludes by selecting the model with fewer parameters as the more challenging and interesting solution. This model strikes a balance between performance and efficiency, and represents a promising direction for future work.

## 📚 References

- Vaswani, A., et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 2017.
- [Transformers by Hugging Face](https://huggingface.co/transformers/)
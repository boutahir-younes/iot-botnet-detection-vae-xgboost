<div align="center">

# IoT Botnet Detection with a Variational Autoencoder (VAE) and XGBoost on the N-BaIoT Dataset

**Hybrid deep learning and machine learning pipeline for detecting Mirai and Gafgyt botnet attacks on real IoT network traffic**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-006400)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Results](#results)
3. [Model Architecture](#model-architecture)
4. [Training Configuration](#training-configuration)
5. [Dataset](#dataset)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Design Choices](#design-choices)
8. [Visualizations](#visualizations)
9. [Getting Started](#getting-started)
10. [Project Structure](#project-structure)
11. [Limitations](#limitations)
12. [Tech Stack](#tech-stack)
13. [References](#references)
14. [License](#license)

---

## Overview

Connected devices such as cameras, doorbells and baby monitors are frequent targets of botnets like Mirai and Gafgyt. This project detects these attacks from network traffic statistics using a two-stage hybrid pipeline:

1. **Variational Autoencoder (VAE).** An unsupervised neural network, trained on benign traffic only, compresses each network flow from 115 features into a compact 20-dimensional latent representation.
2. **XGBoost.** A gradient boosting classifier takes the latent features and predicts one of 11 classes: benign traffic, five Gafgyt attack types and five Mirai attack types.

The pipeline is evaluated on N-BaIoT, a public dataset of more than 7 million real network flows captured from 9 commercial IoT devices.

## Results

### Binary classification (normal versus attack)

| Metric | Score |
|---|---|
| Accuracy | 99.99 % |
| Precision | 99.99 % |
| Recall | 99.99 % |
| F1-score | 99.99 % |

### Multi-class classification (11 classes)

Evaluated on a stratified hold-out test set of 1.41 million flows.

| Metric | Score |
|---|---|
| Accuracy | 87.83 % |
| Precision (weighted) | 93.62 % |
| Recall (weighted) | 87.83 % |
| F1-score (weighted) | 83.66 % |
| Training time (XGBoost) | 3,470 s |
| Prediction time (test set) | 88 s |

Nine of the eleven classes are recognized almost perfectly. The multi-class score is limited mainly by one pair of classes: `gafgyt_tcp` and `gafgyt_udp`. The Kitsune features do not encode the transport layer protocol, so the two attacks have nearly identical statistical signatures and the classifier mostly predicts `gafgyt_udp` for both. This confusion appears in both the autoencoder baseline and the VAE model, which suggests it comes from the data rather than from the model choice.

## Model Architecture

```text
Network flow (115 features)
        |
        v
+-----------------------+
|     VAE encoder       |   115 -> 96 -> 64 -> 32
|                       |
|  mu (20) + log var(20)|   probabilistic bottleneck
|  z = mu + sigma * eps |   reparameterization trick
+-----------------------+
        |
        v   latent features: mu (20 dimensions)
+-----------------------+
|       XGBoost         |   400 trees, sequential boosting
+-----------------------+
        |
        v
  Prediction (11 classes)
```

| Component | Layers | Dimensions |
|---|---|---|
| Encoder | 3 dense layers | 115 -> 96 -> 64 -> 32 |
| Bottleneck (mean) | 1 dense layer | 32 -> mu (20) |
| Bottleneck (variance) | 1 dense layer | 32 -> log sigma^2 (20) |
| Decoder | 3 dense layers | 20 -> 32 -> 64 -> 96 -> 115 |

The VAE loss combines a reconstruction term and a regularization term:

```text
L = MSE(x, x_hat) + beta * KL( N(mu, sigma^2) || N(0, I) )
```

The KL term pushes the latent space toward a smooth Gaussian distribution, which gives the classifier a better organized representation than a standard autoencoder.

## Training Configuration

### Variational Autoencoder

| Parameter | Value |
|---|---|
| Loss | MSE + beta * KL divergence, beta = 0.5 |
| Optimizer | Adam, learning rate 1e-3, gradient clip norm 1.0 |
| Scheduler | ReduceLROnPlateau (patience 4, factor 0.5, minimum learning rate 1e-6) |
| Epochs | up to 60 |
| Batch size | 512 |
| Early stopping | patience 10 |
| Training data | benign traffic only |
| Training loss | decreased from 41.56 to 11.27 |

### XGBoost classifier

| Parameter | Value |
|---|---|
| Input | latent mean `mu`, 20 features |
| `n_estimators` | 400 |
| `max_depth` | 10 |
| `learning_rate` | 0.08 |
| `objective` | `multi:softmax` |
| `num_class` | 11 |

## Dataset

**N-BaIoT** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot).

| Property | Value |
|---|---|
| Flows | 7,062,606 |
| Size | about 7.5 GB (99 CSV files) |
| Features | 115, extracted with Kitsune |
| Devices | 9 commercial IoT devices |
| Classes | 11 (1 benign, 5 Gafgyt, 5 Mirai) |
| Train / test split | stratified 80 / 20 (5.65 M / 1.41 M flows) |

The 115 features are statistical aggregations over sliding time windows (5 s, 3 s, 1 s, 100 ms and 10 ms) describing packet volume, magnitude and correlation. Two devices, the Ennio doorbell and the Samsung webcam, were not infected with Mirai.

## Preprocessing Pipeline

Raw Kitsune features can reach values close to 1e12 during DDoS bursts, which makes neural network training diverge. A robust normalization pipeline solves this problem.

| Step | Operation | Effect |
|---|---|---|
| 1 | Loading | 9 devices, 99 files, 7,062,606 rows |
| 2 | Cleaning | Infinite values converted to NaN and removed |
| 3 | Label encoding | 11 classes mapped to integers 0 to 10 |
| 4 | Stratified split | 80 % training, 20 % test, class proportions preserved |
| 5 | Normalization | Percentile clipping (P1 to P99), signed logarithm, standard scaling, clipping at 5 standard deviations |
| 6 | Validation | Assertion that no NaN or infinite value remains |

After normalization, every feature lies in the range [-5, 5] with a standard deviation close to 1. All statistics are computed on the training set only, so no information leaks from the test set.

## Design Choices

**Why train the VAE on benign traffic only?**
The VAE learns what normal traffic looks like. Traffic that differs from this distribution is reconstructed poorly, which is the classic principle of one-class anomaly detection. The latent space it learns then serves as a compact input for the supervised classifier.

**Why use `mu` instead of `z` as input to XGBoost?**
The sampled vector `z = mu + sigma * epsilon` is random and changes at every forward pass. The mean `mu` is deterministic, so it provides stable and reproducible features.

**Why XGBoost?**
Boosting builds trees sequentially, and each tree corrects the errors of the previous ones. Compared with a Random Forest baseline trained on a standard autoencoder (86.35 % accuracy, 82.04 % weighted F1-score), the VAE with XGBoost improves accuracy by 1.48 points and weighted F1-score by 1.62 points. The trade-off is speed: the baseline predicts roughly 3 times faster (27 s against 88 s on the test set).

## Visualizations

<div align="center">

| Latent space (t-SNE) | Confusion matrix |
|:---:|:---:|
| <img src="images/03_latent_tsne.png" width="400"/> | <img src="images/05_multiclasse.png" width="400"/> |

</div>

The structure of the latent space is consistent with the Gaussian prior imposed by the KL regularization.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/iot-botnet-detection-vae-xgboost.git
cd iot-botnet-detection-vae-xgboost
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download N-BaIoT from the [UCI repository](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) and set the `BASE_PATH` variable in the notebook to the folder containing the CSV files.

### 4. Run the notebook

```bash
jupyter notebook notebooks/VAE_XGBoost.ipynb
```

Training on the full dataset is memory intensive. A machine with at least 16 GB of RAM is recommended.

## Project Structure

```text
iot-botnet-detection-vae-xgboost/
├── notebooks/
│   └── VAE_XGBoost.ipynb        # Full pipeline: preprocessing, VAE, XGBoost, evaluation
├── images/                      # Generated figures
├── models/                      # Saved models (created after training)
├── requirements.txt
├── LICENSE
└── README.md
```

## Limitations

- **Protocol ambiguity.** `gafgyt_tcp` and `gafgyt_udp` cannot be reliably separated with these features, which caps the multi-class score.
- **Random split.** The train and test sets are drawn at random from the same captures. Flows from the same attack session can appear on both sides, so the reported scores may be optimistic compared with a deployment on unseen devices or new capture sessions.
- **Single evaluation.** Results come from one split and one training run. Cross-validation or repeated runs with different seeds would give a more reliable estimate.
- **Known attack families only.** The classifier is trained on Mirai and Gafgyt. Its behavior on other botnet families has not been evaluated.
- **Offline evaluation.** The pipeline was evaluated on recorded data, not on live traffic.

## Tech Stack

- **TensorFlow / Keras** for the VAE
- **XGBoost** for the gradient boosting classifier
- **scikit-learn** for preprocessing, metrics and t-SNE
- **NumPy and Pandas** for data handling
- **Matplotlib and Seaborn** for visualization

## References

1. Y. Meidan et al., "N-BaIoT: Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders," *IEEE Pervasive Computing*, vol. 17, no. 3, pp. 12-22, 2018.
2. Y. Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection," *NDSS*, 2018.
3. D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," *ICLR*, 2014.
4. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *KDD*, 2016.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

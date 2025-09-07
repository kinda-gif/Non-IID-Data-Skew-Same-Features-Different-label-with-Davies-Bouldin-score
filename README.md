# SFDL_DB Library

## Overview

SFDL_DB is a Python library designed to facilitate federated learning experiments by generating datasets with "Same Features, Different Label" (SFDL) skew. This type of non Independent and Identically Distributed (non-IID) data distribution is crucial for simulating real-world scenarios where clients might have similar feature sets but different label distributions. The library employs KMeans clustering and leverages the Davies-Bouldin Score to automatically determine the optimal number of clusters (`k`), ensuring effective data partitioning for robust federated learning research and development.

### Mathematical Definition of SFDL Skew

Same features, different label skew in non-IID data refers to the case where the conditional distribution of labels (P(Y∣X)) varies across different subsets of the data, even though the feature distributions (P(X)) remain consistent. In this scenario, the way labels (Y) are associated with a given feature set (X) is not uniform across subsets.

Mathematically, for any subsets i and j, this can be expressed as:

$P_i (Y|X) \neq P_j (Y|X) \quad \text{for } i \neq j$

This implies that, while the features themselves are similarly distributed across the subsets, the labels corresponding to a particular feature set differ, potentially due to variations in local labeling conventions, sensor biases, or demographic differences in the subsets.

### Davies-Bouldin Score

The Davies-Bouldin (DB) index is a metric used to evaluate the quality of clustering. It is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster scatter to between-cluster separation. A lower Davies-Bouldin index indicates a better clustering, meaning that the clusters are more compact and better separated from each other.

Mathematically, the Davies-Bouldin index is calculated as:

$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$

Where:
- $k$ is the number of clusters.
- $\sigma_i$ is the average distance between each point in cluster $i$ and the centroid of cluster $i$ (a measure of cluster scatter).
- $d(c_i, c_j)$ is the distance between the centroids of cluster $i$ and cluster $j$ (a measure of cluster separation).

In simpler terms, the DB index aims to minimize the ratio of within-cluster distances to between-cluster distances. A smaller DB index suggests that clusters are dense and well-separated, which is desirable for effective data partitioning.

## Features

- **SFDL Data Generation**: Creates datasets exhibiting "Same Features, Different Label" skew, essential for simulating realistic federated learning environments.
- **Optimal K-Means Clustering**: Integrates KMeans clustering with automatic optimal `k` selection using the Davies-Bouldin Score, ensuring efficient and meaningful data partitioning.
- **Federated Learning Compatibility**: Produces data splits suitable for diverse client environments in federated learning setups.
- **Flexible Output**: Saves client-specific datasets as CSV files, allowing for easy integration into various machine learning frameworks.
- **Pythonic Interface**: Offers a clean and intuitive API for seamless integration into existing machine learning workflows.

## Installation

To install SFDL_DB, you can use pip:

```bash
pip install sfdl_db
```

Alternatively, if you have cloned the repository, you can install it from the local directory:

```bash
pip install .
```

## Usage

Here's a basic example of how to use SFDL_DB to generate a dataset with SFDL skew:

```python
import pandas as pd
from SFDL_DB.skewing import same_features_different_label_skew

# Example: Create a dummy dataset
data = {
    'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature_2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Generate SFDL skewed data and save to CSVs
same_features_different_label_skew(df, label_col='label', dataset_name='my_sfdl_data', output_dir='./sfdl_output', max_k=5, k_optimal=None)

# You can also specify k_optimal directly if you know the desired number of clusters:
# same_features_different_label_skew(df, label_col='label', dataset_name='my_sfdl_data_fixed_k', output_dir='./sfdl_output', k_optimal=3)
```

## API Reference

### `same_features_different_label_skew(df, label_col, dataset_name='SFDL_DB', output_dir='.', max_k=10, k_optimal=None)`

This is the primary function for generating SFDL skewed datasets. It performs KMeans clustering and distributes data to simulate "Same Features, Different Label" skew.

- **`df`** (pandas.DataFrame): The input DataFrame containing your dataset. It should include features and a label column.
- **`label_col`** (str): The name of the column in `df` that contains the labels. This column will be dropped before clustering.
- **`dataset_name`** (str, optional): A name identifier for the output CSV files. **Default value is `"SFDL_DB"`**.
- **`output_dir`** (str, optional): The directory where the client-specific CSV files will be saved. If the directory does not exist, it will be created. **Default value is `"."` (current directory)**.
- **`max_k`** (int, optional): The maximum number of clusters to consider when searching for the optimal `k` using the Davies-Bouldin Score. This parameter is only used if `k_optimal` is `None`. **Default value is `10`**.
- **`k_optimal`** (int or None, optional): If provided, this number of clusters will be used directly for KMeans clustering, bypassing the optimal `k` determination process. **Default value is `None`**.

**Returns**:
- **None**: This function does not return any value. It saves the generated client-specific datasets as CSV files in the specified `output_dir` and prints messages indicating the filenames and shapes of the saved dataframes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

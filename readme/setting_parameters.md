[← Back to Main](../readme/readme.md#setting_parameters)

# Setting PREFFECT Parameters
The parameters of PREFFECT are edited from within a configuration file (_~/exp/preffect/\_config.py_) which consist of the parameters that control paths, set training/inference parameters, adjust learning parameters, and apply correction variables, in addition to other modifiers for fine-tuning.

## Input/Output Paths
Here we set variables that control pathing (input, output and code locations).

| Variable               | Value                              | Description                     |
|------------------------|------------------------------------|---------------------------------|
| CODE_DIRECTORY         | _/path/_          | Location of code                                   |
| INPUT_ANNDATA_PATH     | _/path/_          | Path to input AnnData file(s)                      |
| OUTPUT_PATH            | _/path/_          | Path to output folder                              |

`CODE_DIRECTORY` points to your installation of PREFFECT. 

`INPUT_ANNDATA_PATH` directs PREFFECT to your input count data, which should be provided as AnnData structures, separated for training, validation and testing purposes (see [Importing data into PREFFECT](importing.md)).

 `OUTPUT_PATH` indicates the path to where your PREFFECT output should be written.

Example setup of these parameters:
```python
configs = {
    # Path parameters
    'CODE_DIRECTORY' : '/PREFFECT_PATH/', # change to your installation path
    'INPUT_ANNDATA_PATH' : '/PREFFECT_PATH/vignettes/simple/',
    # set to your desired output location
    'OUTPUT_PATH' : '/PATH_TO_OUTPUT/example_simple_dataset/', 
```

## Training / Inference Parameters
Here, we list all changeable parameters that control the derivation of a model. 

<a id="train_param"></a>
### Training/Inference Parameters
| Variable               | Value        | Description                                          |
|------------------------|--------------|------------------------------------------------------|
| task                   | train        | _train_ or _inference_                               |
| type                   | simple       | Model type (_simple_, _single_ or _full_)            |
| model_likelihood       | NB           | _NB_ or _ZINB_                                       |
| dispersion             | gene-sample  | Dispersion mode (_gene_, _gene-sample_ or _gene-batch_)|

`task` allows users to specify whether they would like to perform _training_ or _inference_, where _train_ creates and optimizes a PREFFECT model while _inference_ uses the trained PREFFECT model to make predictions on new data. `type` determines the model type being derived: 

- **simple**
  - Generates a network using only a single target tissue and without an adjacency network. 
- **single**
  - Generates a network which incorporates an adjacency matrix for a single target tissue
- **full**
  - Accepts $\cal{T}$ tissues described by both count matrices and adjacency networks.

_single_ and _full_ models require a sample adjacency matrix. `type` will be reset to _simple_ if the adjacency matrix is not found within your input file. 

`model_likelihood` configures PREFFECT to estimate expression counts by determining the parameters of either the Negative Binomial (NB) or Zero-Inflated Negative Binomial (ZINB) distribution, effectively capturing the statistical properties of gene expression data. We chose these distributions after performing a thorough analysis of public FFPE RNAseq datasets, finding that expressed genes more often than not best fit an NB.

`dispersion` alters how the dispersion parameter of the NB/ZINB ($\theta$) is estimated. $\theta$ can be estimated for each gene, for each gene-sample combination, or for each gene within each batch. It serves as the scale parameter to a Gamma distribution and is computed for each gene
which are used to compute $\omega$, the expected fraction of reads for each gene and cell combination.

If `dispersion` is set to _gene-batch_ but no batch information is provided, the parameter is automatically altered to _gene-sample_ by PREFFECT.

```python
configs = {
    # training parameters      
    'task' : 'train', # 'train' or 'inference' 
    'type' : 'simple',  # 'simple', 'single', 'full'
    'model_likelihood' : 'NB', # 'ZINB' or 'NB'
    'dispersion' : 'gene-sample', # 'gene-sample' or 'gene-batch'
    ...
}
```


## Advanced Parameters
These parameters have been optimized through testing of various datasets and hyperparameter searches. They should not be modified until you get a good working understanding of PREFFECT and its various parts. 

<details>
<summary>Learning Parameters</summary>
<a id="learning_param"></a>
<br>

| Variable               | Value       | Description                                           |
|------------------------|-------------|-------------------------------------------------------|
| epochs                 | 10          | Number of epochs to train                             |
| lr                     | 0.0001      | Initial learning rate                                 |
| weight_decay           | 5e-4        | L2 loss on parameters                                 |
| dropout                | 0.3         | Dropout rate (1 - keep probability)                   |
| alpha                  | 0.2         | For leaky_relu                                        |
| h                      | 8           | Number of attention heads                             |
| r_prime                | 48          | Intermediate hidden dimension                         |
| r                      | 16          | Size of latent space; 2r is size of attention head    |
| calT                   | 1           | Number of tissues                                     |
| clamp_value            | 162754      | Max/min values, exp(12)                               |
| mid_clamp_value        | 22026       | exp(10)                                               |
| lib_clamp_value        | 20          | Clamps lib_size_factors to prevent extreme values.    |
| small_clamp_value      | 20          | Max/min values for log var                            |
| mini_batch_size        | 20          | Number of samples per minibatch                       |

- `calT` allows the user to set the number of tissues being evaluated. If `calT` is set to a value exceeding the number of tissue data files found, then `calT` is altered to the latter value.
- The various `clamp` variables prevent learned parameters from becoming overly large/small during early training steps.
- During training, samples are split into multiple subgroups fed separately into the network to update parameters during training; `mini_batch_size` controls this sample number. If $mini\_batch\_size > N$, then PREFFECT will set $mini\_batch\_size = N$. This parameter has no relation to the technical variable _batch_.
</details>

<details>
<summary>Specifying the Loss Function</summary>
<br>

| Variable                | Value | Description                                             |
|-------------------------|-------|---------------------------------------------------------|
| infer_lib_size          | False | Infer library size                                      |
| batch_centroid_loss     | False | Adjust for batches by centroids                         |

- `infer_lib_size` is a boolean that allows PREFFECT to estimate sample library size, which can lead to better correction of expression counts. If False, the library size is computed from the input count matrix.
- `batch_centroid_loss` attempts to adjust the network to normalize the data relative to the centroid of each batch and reduce batch-to-batch variability. This will be automatically set to _False_ if no _batch_ column is found in the input `obs` table.
</details>

<details>
<summary>Feature Masking</summary>
<a id="feature_masking"></a>
<br>


| Variable                | Value | Description                                                  |
|-------------------------|-------|--------------------------------------------------------------|
| training_strategy       | None  | Can be MCAR, MCAR_2 or None                                  |
| lambda_counts           | 0.0   | Fraction of entries of the expression matrix that are masked |
| lambda_edges            | 0.0   | Fraction of edges in sample adjacency matrix that are masked |

`training_strategy` activates feature masking, which is a technique used primarily to prevent overfitting. Currently, we use the MCAR (Missing Completely At Random) method, though others will be added in the future. The MCAR strategy masks the same fraction of entries the same, while MCAR_2 varies the masking fraction from 0 to the set `lambda`. By randomly setting a subset of input features to zero during training (across both the training and validation datasets), the derived model should better generalize to new, unseen data. 

`lambda_counts` controls the fractions of masking events across all count matrices that will be masked (set to $0$). Both the original and masked count matrices are stored in PREFFECT within the _Rs_ and _Xs_ variables, respectively. If MCAR_2 strategy is used, then this value is the maximum fraction of masking events that occur per gene.

`lambda_edges` controls the fractions of edges within the sample adjacency matrix that will be masked. If MCAR_2 strategy is used, then this value is the maximum fraction of masking events that occur per sample.

Example setup of these parameters:
```python
configs = {
  # Path parameters
  'training_strategy' : "MCAR", # can be MCAR, MAR,  MNAR, or None
  'lambda_counts' : 0.1, # the fraction of all entries of the expression matrix that are masked
  'lambda_edges' : 0.1, # the fraction of all entries of the expression matrix that are masked
  ...
}
```
</details>

<details>
<summary>Correction Variables</summary>
<br>

| Variable               | Value       | Description                                           |
|------------------------|-------------|-------------------------------------------------------|
| correct_vars           | True        | Correct categorical or continuous variables           |
| vars_to_correct        | (Name,Type) | Correction variables; Type (categorical or continuous)|
| adjust_vars            | False       | Whether to adjust for batch during inference step     |
| adjust_to_batch_level  | value       | Set all samples as a certain batch during inference   |

`vars_to_correct` [(Name [str],Type [str])] specifies what sample-based variable should be included into the network during **model training**. The _Name_ entered must match a column within the `obs` table of the AnnData input file (e.g. _batch_). The _Type_ indicates whether the variable input is categorical or continuous. This must be specified by the user. If categorical, the adjustment variable should have levels (e.g. from 0, 1, ..., n-1). Users can specify multiple correction and adjustment variables if desired (e.g. [('batch', 'categorical'), ('age', 'continuous)]). At the moment, PREFFECT requires this information to have no missing data (i.e. no NANs).

`adjust_vars` [Boolean] allows the user to alter the technical variable _batch_ during inference (only possible if _batch_ was included as a correction variable when training the PREFFECT model being used). `adjust_to_batch_level` [Int] indicates what batch value you wish to set all samples to during inference.

Example setup of these parameters:
```python
configs = {
  'correct_vars' : True,
  'vars_to_correct' : [('batch', 'categorical')], #[(var name, type)] where type is either categorical or continuous
  'adjust_vars' : False, 
  'adjust_to_batch_level' : 0,
  ...
}
```

</details>

<details>
<summary>Setting Number of Samples and Genes</summary>

Users can set how many samples or genes should be included during training/inference without having to alter the AnnData input file.
<br>

| Variable                  | Value    | Description                                           |
|---------------------------|----------|-------------------------------------------------------|
| select_samples            | Inf      | Choose a subset of samples in AnnData file randomly   | 
| select_sample_replacement | False    | If select_samples > M, this is automatically True     |
| select_genes              | Inf      | Number of genes in test dataset                       |

- If you wish to use all genes and/or samples in your dataset, set parameters to infinity (_float('inf'_))
- If the user requests more samples than are present in the AnnData input, samples are added by replacement

Example setup of these parameters in `_config.py`:
```python
configs = {
  'select_samples' : float('inf'), # if this is an int M, then samples are randomly selected.  
  'select_sample_replacement' : False, # if select_samples > observed M, this is automatically True.
  'select_genes' : 1000, # if this is an int n, then genes are randomly selected to level n
  ...
}
```
</details>

<details>
<summary>Weighting of Loss Parameters</summary>
<br>

| Variable             | Value               | Description                              |
|----------------------|---------------------|------------------------------------------|
| kl_weight            | 1                                            | Weight for KL-divergence loss               |
| X_recon_weight       | 1                                            | Weight for X reconstruction loss            |
| DA_recon_weight      | [1 for i in range(model_parameters['calT'])] | Weights for DA reconstruction loss          |
| lib_recon_weight     | [1 for i in range(model_parameters['calT'])] | Weight for library size reconstruction loss |
| DA_KL_weight         | [1 for i in range(model_parameters['calT'])] | Weights for DA KL-divergence loss           |
| DL_KL_weight         | [1 for i in range(model_parameters['calT'])] | Weights for DL KL-divergence loss           |
| simple_KL_weight     | [1 for i in range(model_parameters['calT'])] | Weights for simple model KL-divergence loss |
| batch_centroid_weight| [1 for i in range(model_parameters['calT'])] | Weights for batch correction loss           |

The above parameters allow the user to adjust the contribution of each component of the loss. They are combined into a weighted average of the loss. The general equation is as follows:

`(reconstruction losses) + weight*(losses)`

Some losses pertain to individual tissues, while others are a singular loss value. This is why some weights are integers while others are a list of integers across calT (the parameter controlling the number of tissues being run).
</details>

<details>
<summary>Delaying Loss Parameters</summary>

Delay parameters are integers that indicate to PREFFECT at which epoch number should a particular loss value be adjusted for (where an epoch of 1 is the very start of training).
<br>


| Variable               | Value       | Description                                           |
|------------------------|-------------|-------------------------------------------------------|
| delay_kl_lib           | 1           | Delay application of KL loss on library size          |
| delay_kl_As            | 1           | Delay application of KL loss on As                    |
| delay_kl_simple        | 1           | Delay application of KL loss on simple model          |
| delay_recon_As         | 1           | Delay application of reconstruction loss on As        |
| delay_recon_lib        | 1           | Delay application of reconstruction loss on library   |
| delay_recon_expr       | 1           | Delay application of reconstruction loss on expression|
| delay_centroid_batch   | 1           | Delay application of batch correction loss            |

Note: PREFFECT will end with an assertion if all parameters are > 1, as no losses would be applied at all.

</details>
<details>
<summary>Transfer Learning and Miscellaneous Parameters</summary>
<br>

| Variable               | Value       | Description                                           |
|------------------------|-------------|-------------------------------------------------------|
| use_pretrain           | False       | Use pre-trained model                                 |
| PRETRAIN_MODEL_PATH    | _/path/_    | Full path to pre-trained model                        |
| MODEL_FILE             | _model.pth_ | Name of model generated during training               |
| save_model             | True        | Whether to save a model during training               |
| NICK_NAME              | test        | Assignment of nickname for a run                      |
| INFERENCE_FILE_NAME    | inference   | Name of file named when performing inference          |
| logging_interval       | 1           | Batches between log outputs                           |

`use_pretrain` allows you to load a separate model into PREFFECT as a separate starting point. Note that the input data should have the same genes and gene order as the data used to derive the pre-trained model. This is described in more detail in [Transfer Learning](transfer_learning.md).
</details>
<br>

##
[← Back to Main](../readme/readme.md#setting_parameters)

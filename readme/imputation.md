[← Back to Main](../readme/readme.md#imputation)

## Imputation
Imputated count values from a PREFFECT model can be extracted in Python.

To load a pre-trained model into your Python session.
```python
    # Create a factory object where 'output_path' is the path to your PREFFECT model
    preffect = factory(                      
            type='simple', # match the conditions of the model
            ...
            output_path = "/path/to/PREFFECT/Model/" 
    )
    _ = preffect.reinstate(fname=None)
```

One can also train a PREFFECT model within the same Python session, if desired (see [Training a PREFFECT model](training.md)).

The imputed values of the models can then be extracted as an AnnData structure:

```python
inference_object = preffect_object_reinstated.inference_dict['endogenous']
adata_imputed = inference_object.impute_values()
```

This AnnData structure will consist of:

`adata_imputed[0].X` - The estimated count values for all genes and samples (the mean of the estimated NB/ZINB for each gene-sample pair)

`adata_imputed[0].layers["original_counts"]` - Count data used for training

`adata_imputed[0].layers["X_hat_theta"]` - The estimated dispersion of the estimated NB/ZINB for each gene-sample pair

`adata_imputed[0].layers["X_hat_pi"]` - The estimated dropout of the estimated ZINB for each gene-sample pair (will not exist if `model_likelihood` = "NB")

`adata_imputed[0].obs` - Batch and/or Subtype information, if provided.

Note that the AnnData strcuture will be a list, where [0] corresponds to the primary tissue. If using a multi-tissue models, [1] will be the second tissue used, [2] will be the third, etc.

## Evaluating Imputation of Artificially Zeroed Counts

PREFFEECT allows for the intentional masking of certain positions (replacing random counts with artificial zeroes) to simulate missing data during training to evaluate how well a model can impute those missing values. Similarly, PREFFECT allows for random masking of the sample-sample edge matrix (which assist graph attention mechanisms to identify the most informative correlations in the data). 

```python
masking_strategy' : "MCAR", # currently can be "MCAR" or None
'lambda_counts' : 0.1, # the maximum fraction of all entries of the expression matrix that are masked
'lambda_edges' : 0.0, # the maximum fraction of all entries of the sample-sample adjacency matrix that are masked
```

After training is complete, one can use the 'impute_experiment' task to evaluate the accuracy of both the masked positions, and an equal but random sampling of unmasked positions. These accuracy measures are provided as either mean relative error (`error_type="mre"`) or mean square error (`error_type="mse"`).
```python
# error_type can be mre (mean relative error) or mse (mean square error)
inference_object, error_masked, error_unmasked, mre = preffect.impute_experiment(inference_key = 'endogenous', error_type = 'mre')
```

The output from `impute_experiment` consists of:

`inference_object': The inference object (the same as preffect.pr.inference_dict['endogenous'])

`error_masked`: The MRE/MSE between the true and imputed counts of a randomly sampled set of gene-samples pairs that were randomly masked (zeroed)

`error_unmasked: The MRE/MSE between the true and imputed counts of a randomly sampled set of gene-samples pairs that were not randomly masked

`mre`: A table of every gene-sample pair comparison performed by the test:

[← Back to Main](../readme/readme.md#imputation)
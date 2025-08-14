[← Back to Main](../readme/readme.md#inference-with-preffect)

# Performing Inference 
Using a trained model to make predictions or decisions based on unseen test data.


## Performing Interence using the PREFFECT Factory
We can perform Inference through the PREFFECT Factory, a series of classes that can be incorporated into external Python scripts. These tasks include training, inference, reinstatement, clustering, and visualization based on specified configurations.

You must first import preffect_factory, and configuration settings (if not set from within the script itself).
```python
    from preffect_factory import factory
```

Inference uses PREFFECT object (created through training) as input. If training was not performed within your session and you wish to import a previously trained model, we must use the 'reinstate' task.

```python
    # Create a factory object where 'output_path' is the path to your PREFFECT model
    preffect = factory(                      
            type='simple', # match the conditions of the model
            ...
            output_path = "/path/to/PREFFECT/Model/" 
    )
    _ = preffect.reinstate(fname=None)
```


Parameters should remain as they were in training, with the exception of `input_inference_anndata_path` (which can be changed to direct PREFFECT to a test dataset). If this was not set when creating the factory, it can be set directly after the fact:

```python
preffect.configs['input_inference_anndata_path'] = "/path/to/test/"
_ = preffect.inference(inference_key="test_dataset")
```
`inference_key` assigns your inference a name. 

The test dataset is expected to have the same set of genes as the data used for training, in the same order. The number of patients can differ, however (we suggest a 6:2:2 train/validation/test ratio). If you generated a model that utilized a sample adjacency matrix (`type` _single_ or _full_), the test dataset will also require this information.

## Adjusting for Batch
`adjust_vars` allows users to set PREFFECT to treat all samples being passed through the model as if they were derived from a specific batch used during training. The 'batch' variable must be included as a correction variable (through the `correct_vars` and `vars_to_correct` parameters) during PREFFECT model training to perform this task.
```python
preffect.configs['adjust_vars'] = True
preffect.configs['adjust_to_batch_level'] = 0
_ = preffect.inference()
```

If not set, the batch adjustment inference will be named `inference_#` where # is the batch level being adjusted to (0 in this case). The `inference_dict` contains all inference runs, the key of which is set by the `inference_key` parameter (or 'endogenous' in the case of endogenous inference). 
```python
    inference_object = preffect.pr.inference_dict['inference_0'] 
```

Within `inference_object`, there are a series of useful functions. Here, we import the results of Inference as an AnnData structure.
```python
    adata_inf = inference_object.return_counts_as_anndata()
```

This AnnData structure will consist of:

`adata_inf[0].X` - The estimated count values for all genes and samples (the mean of the estimated NB/ZINB for each gene-sample pair)

`adata_inf[0].layers["original_counts"]` - Count data used for training

`adata_inf[0].layers["X_hat_theta"]` - The estimated dispersion of the estimated NB/ZINB for each gene-sample pair

`adata_inf[0].layers["X_hat_pi"]` - The estimated dropout of the estimated ZINB for each gene-sample pair (will not exist if `model_likelihood` = "NB")

`adata_inf[0].obs` - Batch and/or Subtype information, if provided.

Note that the AnnData strcuture will be a list, where [0] corresponds to the primary tissue. If using a multi-tissue models, [1] will be the second tissue used, [2] will be the third, etc.

## Endogenous Inference is Performed Immediately After Training

Endogenous Inference, in which the training data is run through the model, is immediately performed after training by PREFFECT. A description of how to import these results into a Python session are
described in [_Importing and Investigating Inference_](#inf_import). 


This command will create the `inference_object`, as well as create a series of figures within the /inference/ folder (see 'Inference Output'). The `inference_key` is used as the key name to this run within `inference_dict` (see [_Importing and Investigating Inference_](#inf_import)).


## Automatically Generated Inference Images

The Inference object include functions that generate summary plots of the results. However, these are automatically generated when the inference task is completed (see Inference Output for its location).

visualize_lib_size.pdf - Draws histograms for the observed library size per sample, and the library size estimated by the model (the latter only appears if `infer_lib_size` is True).

visualize_libsize_and_dispersion.pdf - Plots illustrating the estimated NB mean and dispersion across all genes.

visualize_batch_adjustment.pdf - Scatterplots comparing the overall average of gene counts between batches (the file will not be generated if batch information is not provided).

visualize_gene_scatterplot.pdf - 50 scatterplots displaying the observed and estimated counts of each sample on an individual gene basis (only the first 50 genes are displayed).

visualize_latent_recons_umap.pdf - Two UMAPs which cluster the latent space and the reconstruction error after Inference. Clusters are coloured according to batch (if provided).


## Inference Output
When a run is complete, you will find new/modified files in your `OUTPUT_PATH`:

```
/OUTPUT_PATH/
│
|── model.pth (updated to include inference data)
|
├── /logs/
│   ├── inference_preffect.log (updated with the details of the inference run)
│
├── inference/
    ├── endogenous/ or inference_[batch no.]/
       ├── visualize_batch_adjustment.pdf
       ├── visualize_gene_scatterplot.pdf
       ├── visualize_latent_recons_umap.pdf
       ├── visualize_lib_size.pdf
       ├── visualize_libsize_and_dispersion.pdf
```

##
[← Back to Main](../readme/readme.md#inference-with-preffect)
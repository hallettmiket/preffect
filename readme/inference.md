[← Back to Main](readme.md#inference-with-preffect)

# Performing Inference 
Using a trained model to make predictions or decisions based on unseen test data.

# Setting `_config.py` for Inference

Once a PREFFECT model has been generated, we can use it to perform inference on the same or a separate test data set. For the purposes of this tutorial, we will perform inference on the training dataset for simplicity.

```python
configs = {
    # Path parameters
    'CODE_DIRECTORY' : '/PREFFECT_PATH/', # change to your installation path
    'INPUT_ANNDATA_PATH' : '/PREFFECT_PATH/exp/preffect/vignettes/simple/', # test data
    'OUTPUT_PATH' : '/PATH_TO_OUTPUT/example_simple_dataset/', # set to your desired output location
   
    # training parameters      
    'task' : 'inference', # train or inference 
    'type' : 'simple',  # 'simple', 'single', 'full'
    'model_likelihood' : 'NB', # ZINB or NB
    ...
    # leave all downstream parameters as default for this tutorial
}
```
Parameters should remain as they were in training, with the exception of `INPUT_ANNDATA_PATH` (which can be changed to direct PREFFECT to a test dataset) and `task` (which should be changed to _inference_). 

The test dataset is expected to have the same set of genes as the data used for training, in the same order. The number of patients can differ, however. If you generated a model that utilized a sample adjacency matrix (`type` _single_ or _full_), the test dataset will also require this information.

## Adjusting for Batch
`adjust_vars` allows users to set PREFFECT to treat all samples being passed through the model as if they were derived from a specific batch used during training.
```python
configs = {
    'adjust_vars' : True, 
    'adjust_to_batch_level' : 1,
    ...
}
```
It is a requirement that 'batch' is included as a correction variable (through the `correct_vars` and 'vars_to_correct' parameters) during PREFFECT model training for this option to function properly.

## Endogenous Inference is Performed Immediately After Training

Endogenous Inference, in which the training data is run through the model, is immediately performed after training by PREFFECT. A description of how to import these results into a Python session are
described in [_Importing and Investigating Inference_](#inf_import). 

## Performing Interence using the PREFFECT Factory
We can perform Inference through the PREFFECT Factory, a series of classes that can be incorporated into external Python scripts. These tasks include training, inference, reinstatement, clustering, and visualization based on specified configurations.

You must first import preffect_factory, and configuration settings (if not set from within the script itself).
```python
    from preffect_factory import factory
    from _config import configs
```

Inference uses PREFFECT object (created through training) as input. If training was not performed within your session and you wish to import a previously trained model, we must use the 'reinstate' task.

```python
    # Point to the PREFFECT model you wish to import
    configs['output_path'] = "/location/of/PREFFECT/Model/" 
    
    preffect_object_reinstated = factory(
        task='reinstate', 
        configs=configs, 
        trigger_setup=True
    )
```

Set the desired task in `_config.py` or within the script itself. In this example, we set inference to treat all samples as they came from a particular batch.

```python
    configs['adjust_vars'] = True
    configs['adjust_to_batch_level'] = 0
```

We then perform the 'inference' task, which creates an Inference PREFFECT object. The model file is also updated with this information.
```python
    inference_object = factory(
        task='inference', 
        configs=configs, 
        preffect_obj=preffect_object_reinstated, 
        inference_key = "inference_name"
    )
```

This command will create the `inference_object`, as well as create a series of figures within the /inference/ folder (see 'Inference Output'). The `inference_key` is used as the key name to this run within `inference_dict` (see [_Importing and Investigating Inference_](#inf_import)).


## Performing Interence through PREFFECT's Command-Line interface

When the `task` parameter in `_config.py` has been set to 'inference', we can run `preffect_cli.py` directly on the command line.
```bash
    $ python ./exp/4_preffect/preffect_cli.py
```

The PREFFECT model will be updated with the results of this task (within `inference_dict`). 


<a id="inf_import"></a>

## Importing and Investigating Inference

The PREFFECT model is updated to include the results of the Inference task performed. To import these results:
```python
    from preffect_factory import factory
    from _config import configs

    # Point to the PREFFECT model you wish to import
    configs['output_path'] = "/location/of/PREFFECT/Model/" 
    
    preffect_object_reinstated = factory(
        task='reinstate', 
        configs=configs, 
        trigger_setup=True
        )
```

The `inference_dict` contains all inference runs, the key of which is set by the `inference_key` parameter (or 'endogenous' in the case of endogenous inference). 
```python
    inference_object = preffect_object_reinstated.inference_dict['endogenous'] 
```

Within `inference_object`, there are a series of useful functions. Here, we import the results of Inference as an AnnData structure.
```python
    adata = inf_reinstate.return_counts_as_anndata()
```


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
[← Back to Main](readme.md#inference-with-preffect)
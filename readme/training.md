[← Back to Main](../readme/readme.md#training-a-preffect-model)

# Training a PREFFECT Model

## Setting `_config.py`

We begin by setting the parameters in `_config.py` for training. PREFFECT provides synthetic example datasets for each type of model that can be generated (`type` _simple_, _single_ and _full_). This is how you would set-up training a _simple_ model.

```python
configs = {
    # Path parameters
    'CODE_DIRECTORY' : '/PREFFECT_PATH/', # change to your installation path
    'INPUT_ANNDATA_PATH' : '/PREFFECT_PATH/vignettes/simple/', # expects subdirectories train and validation
    'OUTPUT_PATH' : '/PATH_TO_OUTPUT/example_simple_dataset/', # set to your desired output location
   
    # training parameters      
    'task' : 'train', # train or inference 
    'type' : 'simple',  # 'simple', 'single', 'full'
    'model_likelihood' : 'NB', # ZINB or NB
    ...
    # leave all downstream parameters as default for this tutorial
}
```

Creating a multi-tissue _full_ model is similar, however you must also alter the `calT` parameter which controls the number of tissues that PREFFECT will include in its network:
```python
configs = {
    # Path parameters
    'CODE_DIRECTORY' : '/PREFFECT_PATH/', # change to your installation path
    'INPUT_ANNDATA_PATH' : '/PREFFECT_PATH/vignettes/full/', # expects subdirectories train and validation
    'OUTPUT_PATH' : '/PATH_TO_OUTPUT/example_full_dataset/', # set to your desired output location
   
    # training parameters      
    'task' : 'train', # train or inference 
    'type' : 'full',  # 'simple', 'single', 'full'
    'model_likelihood' : 'NB', # ZINB or NB

    'calT' : 2, # example data has 2 tissues
    ...
    # leave all downstream parameters as default for this tutorial
}
```
If `calT` remains at its preset of 1, then only the primary tissue would be included for training. If `calT` is set higher than the number of tissue datasets available, it will be automatically set to the number of tissues identified ($2$ in this case).

If you would like to create your own AnnData file to use as PREFFECT input, please refer to [Import and Structure of Data for PREFFECT](importing.md).

## Setup: Batch Correction
PREFFECT can optionally correct for batch, if this information is available in the input AnnData `obs` table:
```python
    'correct_vars' : True,
    'vars_to_correct' : [('batch', 'categorical')], #[(name, type)], where 'type' is either categorical or continuous
    'batch_centroid_loss' : False,
```
The column name within the `obs` table must match `vars_to_correct` ('batch'). 

PREFFECT can correct for any categorical or continuous variable that is provided in the AnnData `obs` table.It can perform this correction across more than one variable at a time. To do so, set `vars_to_correct` as follows:
```python
    'vars_to_correct' : [('batch', 'categorical'), ('misc', 'continuous'), ...],
```

Batch correction can be performed on the example dataset provided, as the synthetic samples have been provided a batch identifier (counts from different batches were artificially given a batch effect, where gene counts from _batch 0_ will generally have lower expression than _batch 1_).

`batch_centroid_loss` attempts to adjust the network to normalize the data relative to the centroid of each batch and reduce batch-to-batch variability. This is optional, and is separate from the variable correction performed to the variables listed in `vars_to_correct`.

## Running PREFFECT for Training using PREFFECT Factory

We can train a PREFFECT model through the PREFFECT Factory, a series of classes that can be incorporated into external Python scripts. These tasks include training, inference, reinstatement, clustering, and visualization based on specified configurations.

To perform training using the PREFFECT Factory, you must first import preffect_factory and configuration settings.
```python
    from preffect_factory import factory
    from _config import configs
```

You can also set your configuration variables from within the Python script itself.
```python
    configs['calT'] = 1
    configs['h'] = 8
    ...
```

To perform training, call `factory()` with task set to "train".
```python
    preffect_object = factory
        (configs=configs.copy(), 
        task='train', 
        always_save = True
    )
```

This generates both a PREFFECT object (for downstream analysis), and saves a PREFFECT model, logs, and a plot illustrating the progression of loss during training (see 'Output'). The PREFFECT object can then be used by other `factory()` tasks.

The "always_save" parameter is set to True by default. If this is turned off, the current state of the process (preffect or inference) will not be saved after execution.

## Running PREFFECT for Training using a Command-Line Interface

We can training a PREFFECT model from the command line. The following command will develop a single PREFFECT model using the parameters listed in `_config.py:`
```bash
$ python ./preffect/preffect_cli.py
```

You can also run PREFFECT with parameters set entirely on the command line:
```bash
python ./preffect/preffect_cli.py --NICK_NAME test_1 --INPUT_ANNDATA_PATH /PREFFECT_PATH/exp/preffect/vignettes/simple/ --epochs 200 --mini_batch_size 400 --lr 0.001 --infer_lib_size True --model_likelihood NB  --batch_centroid_loss False --select_samples 100 --select_sample_replacement True --task inference --adjust_vars False --OUTPUT_PATH /PATH_TO_OUTPUT/example_simple_dataset/ --save_model True
```

You can import the generated PREFFECT model to a Python script using PREFFECT Factory's 'reinstate' task: 
```python
    # Point to the PREFFECT model you wish to import
    configs['output_path'] = "/location/of/PREFFECT/Model/" 

    preffect_object_reinstated = factory(
        task='reinstate', 
        configs=configs, 
        trigger_setup=True
    )
```


## Output
When a run is complete, you will find the results in your `OUTPUT_PATH`. It will consist of the following:

```
/OUTPUT_PATH/
│
├── model.pth (the derived model)
|
|-- /logs/
|   |-- input_preffect.log
│   ├── forward_preffect.log
│   ├── inference_preffect.log
│
├── inference/ (empty until inference is performed)
│
└── results/
    ├── losses.pdf (plot of change in loss parameters during training)
```


##
[← Back to Main](../readme/readme.md#training-a-preffect-model)
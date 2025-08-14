[← Back to Main](../readme/readme.md#transfer-learning)

# Transfer Learning
Utilizing the learned features of a previously trained PREFFECT model to derive a new model.

## Setting `_config.py` for Transfer Learning
Transfer learning allows you to use a previously derived model as a separate starting point for the derivation of a new model using a separate dataset. To perform transfer learning, you must modify the following parameters in `_config.py`.

```python
configs = {
    # Path parameters
    'CODE_DIRECTORY' : '/PREFFECT_PATH/', # change to your installation path
    'INPUT_ANNDATA_PATH' : '/PATH_TO_NEW_TRAINING_DATA/', # test data
    'OUTPUT_PATH' : '/PATH_TO_OUTPUT/NEW_MODEL_OUTPUT/', # set to your desired output location
   
    # training parameters      
    'task' : 'train', # train or inference 
    'type' : 'simple',  # 'simple', 'single', 'full'
    'model_likelihood' : 'NB', # ZINB or NB
   
    # pre-training parameters
    'use_pretrain' : True,
    'PRETRAIN_MODEL_PATH' : '/PATH_TO_MODEL/model.pth', # exact path to the pre-trained model you wish to use
    ...
}
```
Then create a PREFFECT object and use the `train()` command as usual.

When applying transfer learning, it is crucial that the input data used to train the original model aligns with your new training data. The new training data should have the same genes in the same gene order as the data used to derive the pre-trained model. 

Similarly, many training parameters must be unchanged to maintain the general structure of the network. All encoding and decoded layers must be present and in the expected shape. This means the following parameters cannot be changed: `task`, `type`, `infer_lib_size`, `model_likelihood`, `dispersion`, `correct_vars`, `vars_to_correct`, `batch_centroid_loss`, `mini_batch_size`, `calT`, `h`, `r`, and `r_prime`.


### Running PREFFECT for Interence

Now that `_config.py` has been set up, we can begin continue training a model using `preffect_cli`.
```bash
$ python ./preffect/preffect_cli.py
```

Training at this point will continue as normal.

##
[← Back to Main](../readme/readme.md#transfer-learning)
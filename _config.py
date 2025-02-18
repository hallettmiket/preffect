from _utils import ( update_composite_configs )

configs = {
    # Global Parameters
    'code_directory' : '/data/lab_vm/release/preffect/exp/preffect',   
    'input_anndata_path' : '/data/lab_vm/refined/preffect/0_synthetic_data/anndata/main_synthetic/dataset_omega_batch_1/L_Million/',

    'output_path' : '/data/lab_vm/refined/preffect/13_hyper_search/dataset_omega_batch_1/L_Million_rprime128/test',
    'input_existing_session' : 'simple_py_direct', # for reinstating a session, this should be in the output_path directory

    'PREFFECT_version' : '0.1',
    'nick_name' : 'default',
    
    'task' : 'train', ##### train, inference, impute, cluster, reinstate, (developmental: impute_experiment), visualize_*
    'type' : 'simple',  # 'simple', 'single', 'full' <- different type of train tasks.
    
    'mini_batch_size' : 200,  #  num. of samples per minibatch
    'epochs' : 5, # Number of epochs to train
    'lr' : 0.001, # initial learning rate
    'weight_decay' : 5e-4, # L2 loss on parameters.
    'dropout' : 0.1, # dropout rate (1 - keep probability)

    'infer_lib_size' : False,
    'batch_centroid_loss' : False,

    'model_likelihood' : 'NB', # ZINB or NB
    'dispersion' : 'gene-sample', # gene, gene-sample or gene-batch
    'batch_present' : False,  # this is set by the data_loader; do not force.

    'correct_vars' : False,
    'vars_to_correct' : [], #[(var name, type)] where type is either categorical or continuous

    'adjust_vars' : False, 
    'adjust_to_batch_level' : 0,

    'select_samples' : float('inf'), # if this is an int M, then samples are randomly selected.  
    'select_sample_replacement' : False, # if select_samples > observed M, this is automatically True.
    'select_genes' : float('inf'), # if this is an int n, then genes are randomly selected to level n

    'alpha' : 0.2, # for leaky_relu
    'adj_exist' : False,       # whether the objects are found.
    'pos_edge_weight' : 1, # weight of edge for loss function
    'neg_edge_weight' : 1, # weight of non-edge for loss function

    'masking_strategy' : "MCAR", # current options: MCAR or None
    'lambda_counts' : 0.1, # the maximum fraction of all entries of the expression matrix that are masked
    'lambda_edges' : 0.0, # the fraction of all entries of the expression matrix that are masked
 
    'calT' : 1, # number of tissues
    'h' : 8, # number of attention heads
    'r_prime' : 128, # intermediate hidden dimension
    'r' : 20, # size of latent space; also 2r is size of attention head
    'r_embed' : 5, # size of the embedding space for categorical korrection variables.
    
    'logging_interval' : 1, # 1-number of batches between sending output to log file (forward)

    'clamp_value' : 162754,  # correspond to exp(12); max and min values
    'mid_clamp_value' : 22026, # exp(10)
    'lib_clamp_value' : 20, # exp(5)
    'small_clamp_value' : 10,  # for log var; exp(10) ~ 22K; max and min values
    
    'theta_transform' : True, # whether to log transform the theta parameter, improves performance when theta>1
    'gradient_clip' : True, # to perform gradient clipping of a


    # delay of the application of losses (value = epoch)
    'delay_kl_lib' : 1,
    'delay_kl_As' : 1, # delay past total epochs to disable
    'delay_kl_simple': 1,
    'delay_recon_As' : 1,
    'delay_recon_lib' : 1,
    'delay_recon_expr' : 1,
    'delay_centroid_batch' : 1,

    'unit_tests' : False,  # set to True if doing testing.
    'number_padded' : 0,  # set automaticlaly by data loader; do not adjust; tracks "padding samples" during inference

    # for inference
    'inference_overwrite' : True,
    'inference_key' : 'inferential',# expects just a file name without path; this will be saved into INFERENCE_PATH

    # for clustering
    'cluster_file_name' : 'cluster',
    'cluster_overwrite' : True,
    
    'cuda_device_num' : 4,  # from 0:7 on our GPU farm
    'no_cuda' : False,     # Disables CUDA training.
    'seed' : None,  # Random seed or None (don't set seed.)
    'epsilon' : 1e-10,
    # 'cuda_device' : str set automatically by preffect.

    # pre-training
    'use_pretrain' : False,
    'pretrain_model_path' : None,    # expects full path with file name; will copy model to OUTPUT_PATH

    #  models and pretrained models are stored just in OUTPUT_PATH
    'save_training_session' : True,             # True saves it to file


    # upper-trimming 
    'trim_high_expressed_genes' : False,
    'trim_percentage' : 0.01,

}

# The following parameters allow the user to adjust the  contribution of each 
#   component of the loss. They are combined into a weighted average of the loss.
configs['kl_weight'] = 0.1
# weight controlling Reconstruction vs KLDivergence loss
# Next within the reconstruction losses, we can weight their contributions.
# The general formula looks as follows:
# reconstruction = (DA_weight[0]*l(DA^tau1) + DA2_weight[1]*l(DA^tau2) + DA3_weight[2]*l(DA^tau3)) +
#   X_weight* NLL(X_hat)
configs['DA_recon_weight'] = [1 for i in range(configs['calT'])]
# Note the len() == calT
configs['X_recon_weight'] = 20
# 'lib_recon_weight':1,
configs['lib_recon_weight'] = [1 for i in range(configs['calT'])]
# Finally within the KL-diverengce losses, you can weight the Ss versus As:
configs['DA_KL_weight'] = [1 for i in range(configs['calT'])]
configs['X_KL_weight'] = [1 for i in range(configs['calT'])]
configs['DL_KL_weight'] = [1 for i in range(configs['calT'])]
configs['simple_KL_weight'] = [1 for i in range(configs['calT'])]
configs['batch_centroid_weight'] = [1 for i in range(configs['calT'])]

configs['input_inference_anndata_path'] = configs['input_anndata_path'] + 'train/'

configs = update_composite_configs(configs)

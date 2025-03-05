import torch
import os
import logging

from _logger_config import setup_loggers
from _preffect import ( Preffect )
from _inference import( Inference )
from _utils import (
    ensure_directory,
    check_folder_access, 
    check_folder_access,
    update_composite_configs
)
from wrappers._cluster import( Cluster )
from _error import ( PreffectError )

def core_only_copy(old):
    # to save space and for clarity, we remove all non-necessary keys from a dictionary for the config in the subobject ofa  Preffect object (e.g. an Inference object)
    
    remove_list = ['model_state_dict', 'optimizer_state_dict']
    knew = old.copy()
    for key in remove_list:
        knew.pop(key, None)  # None is the default value if key is not found
    return knew

visualizations = [
    ("visualize_lib_size", "visualize_lib_size.pdf"),
    ("visualize_gene_scatterplot", "visualize_gene_scatterplot.pdf"),
    ("visualize_batch_adjustment", "visualize_batch_adjustment.pdf"),
    ("visualize_fraction_pi_per_gene", "visualize_fraction_pi_per_gene.pdf"),
    ("visualize_libsize_and_dispersion", "visualize_libsize_and_dispersion.pdf"),
    ("visualize_latent_recons_umap", "visualize_latent_recons_umap.pdf")
]

possible_inference_visualizations = {
    'visualize_lib_size': Inference.visualize_lib_size, 'visualize_gene_scatterplot': Inference.visualize_gene_scatterplot, 
    'visualize_fraction_pi_per_gene': Inference.visualize_fraction_pi_per_gene, 'visualize_libsize_and_dispersion': Inference.visualize_libsize_and_dispersion,
    'visualize_latent_recons_umap': Inference.visualize_latent_recons_umap, 'visualize_batch_adjustment': Inference.visualize_batch_adjustment}

def setup_cuda(configs):
    r"""
    Set up the CUDA device based on the provided configurations.

    :param configs: A dictionary containing configuration settings. Expected keys include:
        - **'cuda_device_num'**: Integer specifying the CUDA device number to use.
        - **'no_cuda'**: Boolean that if True, forces the use of CPU even if CUDA is available.
    :type configs: dict

    :return: The configured PyTorch device (either CUDA or CPU).
    :rtype: torch.device
    """
    # Set up cuda device if necessary
    cuda_device_number = configs['cuda_device_num']
    configs['cuda_device']=torch.device(
        f'cuda:{cuda_device_number}'
        if torch.cuda.is_available()
        and not configs['no_cuda'] else 'cpu')
    
def generate_and_save_visualizations(inference_instance):
    """
    Function which calls visualization tasks and saves the images within `preffect_factory.py`.

    :param inference_instance: An instance of the Inference class representing the inference process.
    :type inference_instance: Inference

    """

    filepath = os.path.join(inference_instance.configs_inf['inference_path'], inference_instance.configs_inf['inference_key'])

    ensure_directory(filepath)  # this will be a subdirectory of the inference folder.
    for method_name, filename in visualizations:
        visualize_method = getattr(inference_instance, method_name)
        vlib = visualize_method(inference_instance)
        complete_filepath = os.path.join(filepath, filename)
        if vlib is not None:
            inference_instance.save_visualization(vlib=vlib,  filename=complete_filepath)


def factory_setup(configs):
    """
    Sets up the factory environment for processing by initializing paths, ensuring directory 
    existence, and setting up logging and CUDA device configurations.

    :param configs: A dictionary containing configuration settings. Expected keys include:
        
        - **'input_anndata_path'**: Path to input Anndata files for basic checks.
        
        - **'input_inference_anndata_path'**: Path for input inference Anndata files.
        
        - **'output_path'**: Base path where logs and results directories will be created.
        
        - **'cuda_device_num'**: Integer specifying the CUDA device number to use.
        
        - **'no_cuda'**: Boolean that if True, forces the use of CPU even if CUDA is available.
    :type configs: dict

    :return: A tuple containing:
        
        - **input_log** (logging.Logger): Logger for input operations.
        
        - **forward_log** (logging.Logger or None): Logger for forward operations. None if the task is not 'train'.
        
        - **inference_log** (logging.Logger or None): Logger for inference operations. None if the task is not 'train'.
        
        - **configs** (dict): The updated configurations dictionary.
    :rtype: tuple

    :raises Exception: If there is an issue accessing the required Anndata paths, an exception is raised with a message indicating the inaccessible path.

    """

    configs = update_composite_configs(configs)

    try:
        check_folder_access(configs['input_anndata_path'])
    except Exception as e:
        print(e)
  
    ensure_directory(configs['output_path'])
    ensure_directory(configs['log_path'])
    ensure_directory(configs['inference_path'])
    ensure_directory(configs['results_path'])

    if configs['task']=='train':
        input_log, forward_log, inference_log = setup_loggers(configs)
    else:
        forward_log = None
        inference_log = None
    input_log = logging.getLogger('input')
    configs['cuda_device_number'] = setup_cuda(configs)
    input_log.info(f"Cuda available: {torch.cuda.is_available()} Device: { str(configs['cuda_device'])}")

    return input_log, forward_log, inference_log, configs

def factory(
        task=None, 
        always_save=True, 
        trigger_setup=False, 
        configs=None, 
        preffect_obj=None, 
        inference_obj=None, 
        inference_key=None, 
        fname=None, 
        visualize=True,
        error_type=None,
        cluster_omega = False):
    """
    Calls _preffect class to perform tasks such as training, inference, reinstatement, clustering, and visualization 
    based on specified configurations.

    :param task: The specific task to be executed. Valid options include 'train', 'inference', 'reinstate',
                 'cluster_latent', 'cluster_counts', and 'visualize_lib_size'.
    :type task: str, optional
    :param always_save: If True, the current state of the process (Preffect or Inference) will always be saved after execution.
    :type always_save: bool, optional
    :param trigger_setup: If True, triggers the setup process for the environment before execution.
    :type trigger_setup: bool, optional
    :param configs: Configuration dictionary specifying details such as paths, device settings, and the specific task to be executed.
    :type configs: dict, optional
    :param preffect_obj: An instance of the Preffect class, if already instantiated; otherwise, it is created based on the task requirements.
    :type preffect_obj: Preffect, optional
    :param inference_obj: An instance of the Inference class, if already instantiated.
    :type inference_obj: Inference, optional
    :param inference_key: The name identifier for an Inference instance to be fetched from a register.
    :type inference_key: str, optional
    :param fname: The filename to be used for saving outputs, specifically in inference tasks.
    :type fname: str, optional
    :param visualize: If True, enables visualization for relevant tasks.
    :type visualize: bool, optional
    :param error_type: The type of error to be used for error handling and reporting.
    :type error_type: str, optional
    :param cluster_omega: If True, enables clustering of omega values.
    :type cluster_omega: bool, optional

    :return: Depending on the task specified in `configs['task']`:
        - Preffect instance for 'train', 'inference', and 'reinstate' tasks.
        - Cluster instance for 'cluster_latent' and 'cluster_counts' tasks.
        - Visualization object for 'visualize_lib_size' task.
    :rtype: Union[Preffect, Cluster, Visualization]

    :raises PreffectError: If an invalid task is specified or if required resources (like Preffect or Inference instances)
                           are not provided or found for the requested operation.
    """

    pr, ir, ir_name = preffect_obj, inference_obj, inference_key
    configs = update_composite_configs(configs)

    if configs is not None:
        configs = core_only_copy(configs)  # this strips (big) model information.
        
    if task == 'train':
        configs['task'] = task
        input_log, forward_log, inference_log, configs = factory_setup(configs)
        
        forward_log = setup_loggers(configs)
        forward_log = logging.getLogger('forward')

        # If pr is not provided, create a new Preffect instance
        if pr is None:
            pr = Preffect(forward_log, existing_session=False, configs=configs)
            
        pr.train(forward_log)
        if always_save:
            pr.save()

        # changed to pr.configs to ensure transfer of config changes by "sanity checks" 
        configs_inf = pr.configs.copy()
        configs_inf['task'] = 'inference'

        check_folder_access(configs_inf['input_inference_anndata_path'])
        inference_instance = Inference(pr, task='inference', inference_key = configs_inf['inference_key'], configs=configs_inf)
        inference_instance.run_inference()
        inference_instance.configs_inf['inference_key'] = 'endogenous'
        inference_instance.register_inference_run()

        
        if always_save:
            pr.save() # the endogenous inference is saved

        # generate plots of results from endogenous inference
        if visualize:
            generate_and_save_visualizations(inference_instance)
    
        return pr

    elif task == 'inference':
        check_folder_access(configs['input_inference_anndata_path'])
        
        if trigger_setup: 
            input_log, forward_log, inference_log, configs = factory_setup(configs)
        else:
            forward_log = logging.getLogger('forward')

        inference_log = setup_loggers(configs)
        if configs['adjust_vars']:
            if fname is None:
                fname = 'inference_' + str(configs['adjust_to_batch_level'])
        else:
            fname = 'endogenous'
        inference_log = logging.getLogger(fname)

        # If pr is not provided, create a new Preffect instance from file
        if pr is None:  
            pr = Preffect(forward_log, existing_session=True, configs=configs)
        configs_inf = configs.copy()
        configs_inf['inference_key'] = fname
        inference_instance = Inference(task='inference', preffect_obj = pr, configs=configs_inf)
        inference_instance.run_inference()
        inference_instance.register_inference_run()

        if always_save:
            pr.save()

        if visualize:
            generate_and_save_visualizations(inference_instance)
    
        return pr
    
    elif task == 'impute_experiment':
        if pr is None:  
            raise PreffectError('Preffect object must be assigned and the setup triggered already.')

        if inference_key is None:
            raise PreffectError(f'You must provide the key for the inference object: {pr.inference_dict.keys()}')

        # if 'masking_strategy' is None or lambda is zero, this function shouldn't run
        if pr.configs['masking_strategy'] is None or pr.configs['lambda_counts'] == 0:
            raise PreffectError("Task 'impute_experiment' cannot be performed if no values are masked.")
            

        configs = pr.configs
        configs['task'] = 'impute_experiment'      
        infy = Inference(task=task, preffect_obj=pr, configs=configs, inference_key = inference_key)    
        infy.run_inference()

        # plot imputation inference object, as it has masking (endogenous inference does not)
        infy.configs_inf['inference_key'] = 'impute_experiment'
        infy.register_inference_run()
        if visualize:
            generate_and_save_visualizations(infy)


        error_masked, error_unmasked, mse_error = infy.calculate_imputation_error(error_type=error_type)

        return infy, error_masked, error_unmasked, mse_error
        
    
    elif task == 'impute':
        if pr is None:
            raise PreffectError("Must specify a preffect object containing the inference object as the target for imputation.")
        if ir is None:
            ir = pr.find_inference_in_register(ir_name) 
            if ir is None:
                raise PreffectError("Must specify a valid inference object in the Preffect object for imputation.")
        return ir.impute_values()
    

    elif task == 'reinstate':
        if trigger_setup: 
            input_log, forward_log, inference_log, configs = factory_setup(configs)
            
        if fname is not None:
            configs['input_existing_session'] = fname
        return Preffect(inference_log, existing_session=True, configs=configs) # uses configs[input_existing_session]

    elif task == 'cluster_latent':
        if pr is not None:
            if ir is None and ir_name is not None:
                ir = pr.find_inference_in_register(ir_name) 
            if ir is not None:
                cl = Cluster(infer_obj=ir, configs_cluster=configs )
                cl.cluster_latent_space(color_by = "leiden")
                cl.register_cluster()
                if always_save:
                    pr.save()
                return cl 
            else:
                raise PreffectError(f"Did not find registered Inference object {ir_name}.")
        else:
            raise PreffectError("For lib visualization, either an inference object must be provided, or a preffect object along \
                                with the name of a valid inference object registered to in the Preffect object must be specified.")

    elif task == 'cluster_counts':
        if pr is not None:
            if ir is None and ir_name is not None:
                ir = pr.find_inference_in_register(ir_name) 
            if ir is not None:
                cl = Cluster(infer_obj=ir, configs_cluster=configs )
                cl.cluster_counts(cluster_omega=cluster_omega)
                cl.register_cluster()
                if always_save:
                    pr.save()
                return cl 
            else:
                raise PreffectError(f"Did not find registered Inference object {ir_name}.")
        else:
            raise PreffectError("For lib visualization, either an inference object must be provided, or a preffect object along \
                                with the name of a valid inference object registered to in the Preffect object must be specified.")
    elif task == 'cluster_true_counts':
        if pr is not None:
            if ir is None and ir_name is not None:
                ir = pr.find_inference_in_register(ir_name) 
            if ir is not None:
                cl = Cluster(infer_obj=ir, configs_cluster=configs )
                cl.cluster_true_counts()
                cl.register_cluster()
                if always_save:
                    pr.save()
                return cl 
            else:
                raise PreffectError(f"Did not find registered Inference object {ir_name}.")
        else:
            raise PreffectError("For lib visualization, either an inference object must be provided, or a preffect object along \
                                with the name of a valid inference object registered to in the Preffect object must be specified.")
    
    # all Inference visualizations start with "visualize"
    elif task == 'visualize_inference_all':
        if pr is not None:
            if ir is None and ir_name is not None:
                ir = pr.find_inference_in_register(ir_name)
                print(ir.configs_inf['output_path'])
            if ir is not None:
                generate_and_save_visualizations(ir)
            else:
                raise PreffectError(f"Did not find registered Inference object {ir_name}.")
        else:
            raise PreffectError("For visualization, either an inference object must be provided, or a preffect object along \
                                with the name of a valid inference object registered to in the Preffect object must be specified.")
        
    elif task.startswith('visualize'):
        if pr is not None:
            if ir is None and ir_name is not None:
                ir = pr.find_inference_in_register(ir_name)
            if ir is not None:
                vlib = possible_inference_visualizations[task](pr, ir)
                if always_save:
                    file = task + ".pdf"
                    ir.save_visualization(vlib=vlib, configs=configs, filename=file)
                return vlib
            else:
                raise PreffectError(f"Did not find registered Inference object {ir_name}.")

        else:
            raise PreffectError("For visualization, either an inference object must be provided, or a preffect object along \
                                with the name of a valid inference object registered to in the Preffect object must be specified.")


    else:
        raise PreffectError('Task not recognized.')

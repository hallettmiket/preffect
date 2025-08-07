import torch
import os
import logging

from preffect._logger_config import setup_loggers
from preffect._preffect import ( Preffect )
from preffect._inference import( Inference )
from preffect._utils import (
    ensure_directory,
    check_folder_access, 
    check_folder_access,
    update_composite_configs
)
from preffect._config import configs 

from preffect.wrappers._cluster import( Cluster )

from preffect._error import ( PreffectError )

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
    # Set up cuda device if necessary
    cuda_device_number = configs['cuda_device_num']
    configs['cuda_device']=torch.device(
        f'cuda:{cuda_device_number}'
        if torch.cuda.is_available()
        and not configs['no_cuda'] else 'cpu')
    
def generate_and_save_visualizations(inference_instance):
    """
    Function which calls visualization tasks and saves the images within `preffect_factory.py`.
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

    Args:
        configs (dict): A dictionary containing configuration settings. Expected keys include:
            - 'input_anndata_path': Path to input Anndata files for basic checks.
            - 'input_inference_anndata_path': Path for input inference Anndata files.
            - 'output_path': Base path where logs and results directories will be created.
            - 'cuda_device_num': Integer specifying the CUDA device number to use.
            - 'no_cuda': Boolean that if True, forces the use of CPU even if CUDA is available.

    Returns:
        tuple: A tuple containing:
            - cuda_device_number (int): The CUDA device number from the configs.
            - input_log (Logger): Logger for input operations.
            - forward_log (Logger): Logger for forward operations.
            - inference_log (Logger): Logger for inference operations.
            - configs (dict): The updated configurations dictionary.

    Raises:
        Exception: If there is an issue accessing the required Anndata paths, an exception is raised
            with a message indicating the inaccessible path.
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

    #print(f"factory set up task: {configs['task']}")
    if configs['task']=='train':
        input_log, forward_log, inference_log = setup_loggers(configs)
    else:
        forward_log = None
        inference_log = None
    input_log = logging.getLogger('input')
    configs['cuda_device_number'] = setup_cuda(configs)
    input_log.info(f"Cuda available: {torch.cuda.is_available()} Device: { str(configs['cuda_device'])}")

    return input_log, forward_log, inference_log, configs


class factory:
    def __init__(
        self,
        always_save=True,
        trigger_setup=False,
        preffect_obj=None,
        inference_obj=None,
        inference_key=None,
        fname=None,
        visualize=True,
        error_type=None,
        **kwargs
    ):
        
        self.configs = configs
        self.configs.update(kwargs)

        # store args
        self.always_save   = always_save
        self.trigger_setup = trigger_setup
        self.configs       = update_composite_configs(self.configs)
        self.pr            = preffect_obj
        self.ir            = inference_obj
        self.ir_key        = inference_key
        self.fname         = fname
        self.visualize     = visualize
        self.error_type    = error_type

        # strip out heavy model info if needed
        if self.configs is not None:
            self.configs = core_only_copy(self.configs)

        # Optionally run the folder/log/CUDA setup immediately
        if self.trigger_setup:
            self._setup_factory_env()

    def _setup_factory_env(self):
        """
        Run your existing factory_setup(...) to
         - ensure directories exist
         - configure loggers
         - pick a CUDA device
         - return updated configs
        """
        input_log, forward_log, inference_log, cfgs = factory_setup(self.configs)
        # overwrite self.configs with any changes factory_setup made
        self.configs        = cfgs
        self.input_log      = input_log
        self.forward_log    = forward_log
        self.inference_log  = inference_log



    def train(self):
        # if user didn’t trigger setup in __init__, do it now
        if not hasattr(self, 'input_log'):
            self._setup_factory_env()


        self.configs['task'] = 'train'

        
        forward_log = setup_loggers(self.configs)
        forward_log = logging.getLogger('forward')

        # If pr is not provided, create a new Preffect instance
        if self.pr is None:
            self.pr = Preffect(forward_log, existing_session=False, configs=self.configs)
        self.pr.train(forward_log)
        if self.always_save:
            self.pr.save()

        # then automatically do inference after training
        configs_inf = self.pr.configs.copy()
        configs_inf['task'] = 'inference'
        check_folder_access(configs_inf['input_inference_anndata_path'])
        self.ir = Inference(
            self.pr,
            task='inference',
            inference_key=configs_inf['inference_key'],
            configs=configs_inf
        )
        self.ir.run_inference()
        self.ir.configs_inf['inference_key'] = 'endogenous'
        self.ir.register_inference_run()
        if self.visualize:
            generate_and_save_visualizations(self.ir)

        return self.pr


    def inference(self, inference_key='endogenous'):
        """
        Run an inference pass.  
        Returns the Preffect instance (pr), after running inference and optional save/visualize.
        """
        # Ensure environment is set up
        if not hasattr(self, 'input_log') or self.trigger_setup:
            self._setup_factory_env()

        # Check for the existance of the input folder
        path = self.configs['input_inference_anndata_path']
        check_folder_access(path)

        forward_log = self.forward_log or logging.getLogger('forward')

        # (Re)configure inference logger
        setup_loggers(self.configs)   # registers handlers
        if self.configs.get('adjust_vars', False):
            fname = self.fname or f"inference_{self.configs['adjust_to_batch_level']}"
        else:
            fname = inference_key
        inference_log = logging.getLogger(fname)
        self.inference_log = inference_log

        # If pr is not provided, create a new Preffect instance from file
        if self.pr is None:
            self.pr = Preffect(forward_log,
                                existing_session=True,
                                configs=self.configs)

        configs_inf = self.configs.copy()
        configs_inf['inference_key'] = fname

        # Run inference
        self.ir = Inference(
            task='inference',
            preffect_obj=self.pr,
            configs=configs_inf
        )
        self.ir.run_inference()
        self.ir.register_inference_run()

        if self.always_save:
            self.pr.save()
        if self.visualize:
            generate_and_save_visualizations(self.ir)

        return self.pr


    def impute_experiment(self, inference_key=None, error_type=None):
        """
        Perform masked‐value imputation and compute errors.
        Returns:
            infy (Inference): the inference object used for imputation
            error_masked (array): errors on masked entries
            error_unmasked (array): errors on unmasked entries
            mse_error (float): overall MSE
        """
        # Must have run setup and trained/inferred previously
        if self.pr is None:
            raise PreffectError(
                "Preffect object must be assigned and setup triggered already."
            )

        # Determine which inference_key to use
        inference_key = inference_key or self.ir_key
        if inference_key is None:
            valid_keys = list(self.pr.inference_dict.keys())
            raise PreffectError(
                f"You must provide inference_key; available keys: {valid_keys}"
            )

        # Masking must have been used during training to perform imputing
        cfg = self.pr.configs
        if cfg.get('masking_strategy') is None or cfg.get('lambda_counts', 0) == 0:
            raise PreffectError(
                "Task 'impute_experiment' cannot run if no values are masked."
            )

        cfg['task'] = 'impute_experiment'
        cfg_inf = cfg.copy()
        cfg_inf['inference_key'] = inference_key

        # Run the inference pass for imputation
        infy = Inference(
            task='impute_experiment',
            preffect_obj=self.pr,
            configs=cfg_inf,
            inference_key=inference_key
        )
        infy.run_inference()

        infy.configs_inf['inference_key'] = 'impute_experiment'
        infy.register_inference_run()
        if self.visualize:
            generate_and_save_visualizations(infy)

        # Compute imputation errors
        error_masked, error_unmasked, mse_error = (
            infy.calculate_imputation_error(error_type=self.error_type)
        )

        return infy, error_masked, error_unmasked, mse_error

    def impute(self, inference_key=None):
        """
        Perform value imputation using an existing Inference instance.

        If self.ir is already set, it will be used. Otherwise, we try to
        look up the Inference object from self.pr by key.
        Returns ir.impute_values()
        """
        # Preffect instance must have at least one registered inference
        if self.pr is None:
            raise PreffectError(
                "Must specify a Preffect object containing an Inference run for imputation."
            )

        # Determine which key to use
        key = inference_key or self.ir_key
        if key is None:
            valid = list(self.pr.inference_dict.keys())
            raise PreffectError(
                f"You must provide an inference_key; available keys: {valid}"
            )

        # Fetch or reuse the Inference object
        ir = self.ir
        if ir is None or getattr(ir, "configs_inf", {}).get("inference_key") != key:
            ir = self.pr.find_inference_in_register(key)
            if ir is None:
                raise PreffectError(
                    f"No registered Inference found for key='{key}'."
                )
            self.ir = ir

        return ir.impute_values()

    def reinstate(self, fname=None):
        """
        Load an existing Preffect session from disk.

        Args:
            fname (str, optional): the name or path of the existing session to load.
                                   If provided, this will override
                                   configs['input_existing_session'].

        Returns:
            Preffect: a new Preffect instance loading the saved session.
        """
        # 1. Run factory setup if requested
        if self.trigger_setup:
            self._setup_factory_env()

        # 2. If the user passed a filename, tell configs which session to load
        if fname is not None:
            self.configs['input_existing_session'] = fname

        # 3. Instantiate Preffect in existing-session mode
        pr = Preffect(
            self.inference_log,       # or self.forward_log if more appropriate
            existing_session=True,
            configs=self.configs
        )

        # 4. Store on self so later tasks (inference, impute, etc.) can reuse it
        self.pr = pr
        return pr

    def visualize_embedding(self,
                        mode: str,
                        ir=None,
                        ir_name=None,
                        cluster_omega = False,
                        color_by: str = "leiden",
                        perform_leiden=False,
                        cluster_aim = 5) -> Cluster:
        """
        Visualize clustering for samples using UMAP.

        Args:
            mode: one of "latent", "counts" or "true_counts"
            ir:      an existing Inference instance (optional)
            ir_name: key to look up an Inference in self.pr (if ir is None)
            color_by: leiden, louvain, etc.

        Returns:
            A Cluster object
        """
        # We select which type of clustering we want to do
        valid = {"latent", "counts", "true_counts"}
        if mode not in valid:
            raise PreffectError(f"Unknown mode '{mode}'.  Choose one of {valid}.")

        # Make sure the Preffect object exists
        if self.pr is None:
            raise PreffectError(
                "You must have a Preffect instance (self.pr) before clustering."
            )

        # Resolve the Inference object
        if ir is None and ir_name is not None:
            ir = self.pr.find_inference_in_register(ir_name)
        if ir is None:
            raise PreffectError(
                f"Did not find registered Inference object '{ir_name}'."
            )

        # Build the Cluster and dispatch to the chosen method
        cluster = Cluster(infer_obj=ir, configs_cluster=self.configs)
        if mode == "latent":
            cluster.cluster_latent_space(color_by=color_by, perform_leiden=perform_leiden, cluster_aim=cluster_aim)
        elif mode == "counts":
            cluster.cluster_counts(cluster_omega = cluster_omega, perform_leiden=perform_leiden, cluster_aim=cluster_aim)
        else:  # mode == "true_counts"
            cluster.cluster_true_counts(perform_leiden=perform_leiden, cluster_aim=cluster_aim)

        cluster.register_cluster()
        if self.always_save:
            self.pr.save()

        self.cluster = cluster
        return cluster

    def generate_embedding(self,
                        mode: str,
                        ir=None,
                        ir_name=None,
                        cluster_omega = False,
                        color_by: str = "leiden",
                        perform_leiden=False,
                        cluster_aim = 5) -> Cluster:
        """
        Generate an embedding using Leiden clustering.

        Args:
            mode: one of "latent", "counts" or "true_counts"
            ir:      an existing Inference instance (optional)
            ir_name: key to look up an Inference in self.pr (if ir is None)
            color_by: leiden, louvain, etc.

        Returns:
            A Cluster object
        """
        # We select which type of clustering we want to do
        valid = {"latent", "counts", "true_counts"}
        if mode not in valid:
            raise PreffectError(f"Unknown mode '{mode}'.  Choose one of {valid}.")

        # Make sure the Preffect object exists
        if self.pr is None:
            raise PreffectError(
                "You must have a Preffect instance (self.pr) before clustering."
            )

        # Resolve the Inference object
        if ir is None and ir_name is not None:
            ir = self.pr.find_inference_in_register(ir_name)
        if ir is None:
            raise PreffectError(
                f"Did not find registered Inference object '{ir_name}'."
            )

        # Build the Cluster and dispatch to the chosen method
        cluster = Cluster(infer_obj=ir, configs_cluster=self.configs)
        if mode == "latent":
            cluster.cluster_latent_space(color_by=color_by, perform_leiden=perform_leiden, cluster_aim=cluster_aim, draw=False)
        elif mode == "counts":
            cluster.cluster_counts(cluster_omega = cluster_omega, perform_leiden=perform_leiden, cluster_aim=cluster_aim, draw=False)
        else:  # mode == "true_counts"
            cluster.cluster_true_counts(perform_leiden=perform_leiden, cluster_aim=cluster_aim, draw=False)

        cluster.register_cluster()
        if self.always_save:
            self.pr.save()

        self.cluster = cluster
        return cluster
    
    def visualize_inference_all(self, inference_key: str = None):
        """
        Re‐generate and save *all* visualizations for a registered Inference run.

        Args:
            inference_key: the name of an existing Inference in self.pr.inference_dict.
                           If omitted, uses self.ir_key.
        """
        if self.pr is None:
            raise PreffectError(
                "A Preffect object must exist before running visualization."
            )

        # Resolve Inference
        key = inference_key or self.ir_key
        ir = self.ir
        if ir is None and key is not None:
            ir = self.pr.find_inference_in_register(key)
        if ir is None:
            raise PreffectError(
                f"Did not find registered Inference object '{key}'."
            )
        self.ir = ir

        print(f"Writing visualizations to: {ir.configs_inf.get('output_path')}")

        generate_and_save_visualizations(ir)

    def visualize(self,
                  task: str,
                  ir=None,
                  ir_name: str = None) :
        """
        Run any inference visualization by name, e.g. "visualize_umap",
        "visualize_heatmap", etc., using the mapping in
        possible_inference_visualizations.
        """
        if not task.startswith("visualize_"):
            raise PreffectError(f"Task '{task}' is not a visualization task")

        # Ensure we have a Preffect instance and resolve the Inference object
        if self.pr is None:
            raise PreffectError(
                "Need a Preffect instance (self.pr) to visualize."
            )

        if ir is None and ir_name is not None:
            ir = self.pr.find_inference_in_register(ir_name)
        if ir is None:
            raise PreffectError(
                f"Did not find registered Inference object '{ir_name}'."
            )
        self.ir = ir

        # Dispatch to the correct visualization function
        try:
            viz_fn = possible_inference_visualizations[task]
        except KeyError:
            raise PreffectError(f"No visualization registered under '{task}'")

        vlib = viz_fn(self.pr, ir)

        if self.always_save:
            filename = f"{task}.pdf"
            ir.save_visualization(vlib=vlib,
                                  configs=self.configs,
                                  filename=filename)

        return vlib

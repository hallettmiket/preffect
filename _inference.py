import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import anndata as ad
import seaborn as sns
from tqdm import tqdm

import pandas as pd
import umap
from matplotlib.colors import ListedColormap
from torch import Tensor

# this package kept printing a "seed" message I want to avoid
from _distributions import (
    NegativeBinomial
)    

from _utils import (
    To
)
from _data_loader import FFPE_dataset
from _error import ( PreffectError )

class Inference:
    def __init__(self, preffect_obj=None, task='inference', inference_key=None, inference_overwrite=True, configs=None):
        """
        Initializes an Inference instance for processing and analyzing inference runs.

        Args:
            preffect_obj: The parent instance, typically a model or manager that holds or interacts with multiple inference instances.
            configs (dict): Configuration settings for the inference run, which includes various operational parameters 
                            such as 'task', 'cuda_device', and 'inference_key'.
        """
        if configs is None:
            raise PreffectError('No configuration file (configs) passed to Inference.')
        if preffect_obj is None:
            raise PreffectError('No Preffect object provided for Inference to be performed.')
        
        self.parent = preffect_obj
        self.Pis = self.parent.Pis
        self.configs_inf = configs
        if inference_key is not None:
            self.configs_inf['inference_key'] = inference_key
        self.configs_inf['inference_overwrite'] = inference_overwrite

        self.output = None 
        self.clusters = {}

        if task=='inference':
            self.ds = FFPE_dataset(self.configs_inf.copy(), learning_type = 'inference', parent = self)
        # developmental code for testing imputation capacity
        elif task=='impute_experiment':
            self.ds = FFPE_dataset(self.configs_inf.copy(), learning_type = 'impute_experiment', parent = self)

        self.ds.to(f'cuda:{configs["cuda_device_num"]}' if torch.cuda.is_available() and not configs["no_cuda"] else 'cpu')

    def register_inference_run(self):
        """
        Register the current inference instance within the parent object's inference dictionary.

        This method checks if an inference instance with the same name already exists in the parent's
        inference dictionary. If it does not exist or if the overwrite permission is granted, the
        current inference instance is added to the dictionary using the specified inference key.
        If an instance with the same name already exists and overwrite permission is not granted,
        a `PreffectError` is raised.

        :raises PreffectError: If an inference object with the same name already exists in the parent's
                            dictionary and overwrite permission is set to False.

        .. note::
            - The parent object is assumed to be a Preffect object that has an `inference_dict` attribute.
            - The `configs_inf` attribute of the current instance is used to determine the inference key
            and overwrite permission.
            - The current inference instance is copied using `copy.copy()` before being added to the
            parent's dictionary to avoid unintended modifications.
        """
        # This will register the inference instance in the parent which is a Preffect object
        if self.configs_inf['inference_key'] not in self.parent.inference_dict.keys() or self.configs_inf['inference_overwrite']:
            self.parent.inference_dict[self.configs_inf['inference_key']] = copy.copy(self)
        else:
            raise PreffectError(f"Inference objected named {self.configs_inf['inference_key']} already exists and overwrite permission is False.")


    def run_inference(self):
        """
        Executes the inference process using the model configured in the parent instance. It handles different types
        of models including 'full', 'single', and 'simple'. It calculates and aggregates results from the model's output
        across all mini-batches.

        Args:
            None
        """
        start_time = time.time()
        self.parent.model.to(f'cuda:{self.configs_inf["cuda_device_num"]}' if torch.cuda.is_available() and not self.configs_inf["no_cuda"] else 'cpu')
        self.parent.model.eval()
        results = []
        with torch.set_grad_enabled(False): 
            if self.configs_inf['type'] in ('full', 'single'):
                
                batches = self.ds.prep_batch_iterator(trivial_batch=True)
                
                for batch_idx in range(len(batches['X_batches'])):
                    batch = To((f'cuda:{self.configs_inf["cuda_device_num"]}' if torch.cuda.is_available() and not self.configs_inf["no_cuda"] else 'cpu'), self.parent.extract_batch(batches, batch_idx))

                    results.append( self.parent.model(batch) )
                    
            elif self.configs_inf['type']=='simple':
                batches = self.ds.prep_batch_iterator(trivial_batch=True)
                batch_idx = 0
                batch = To((f'cuda:{self.configs_inf["cuda_device_num"]}' if torch.cuda.is_available() and not self.configs_inf["no_cuda"] else 'cpu'),  self.parent.extract_batch(batches, batch_idx))

                results=self.parent.model(batch)

        self.output = self.reconstruct_from_minibatches(results, batches['idx_batches']) # sets X_hat_mu, X_hat_theta, Z_Ls, Z_simples, lib_size_factors, Z_As, Z_Xs, X_hat_pi
 

    def impute_values(self):
        """
        Impute missing values in the input data using the trained model.

        This method uses the simplest form of imputation by returning the reconstructed counts
        as an AnnData object. The imputed values are obtained from the model's output after
        running inference on the input data.

        :return: An AnnData object containing the imputed counts.
        :rtype: AnnData

        :raises PreffectError: If the inference has not been computed before calling this method.
        """
        if self.output is None:
            raise PreffectError(f"Inference object has not been computed.")
        # this is the simplest form of imputation. more to come.
        return self.return_counts_as_anndata().copy() # use parent's anndata for ease


    def calculate_imputation_error(self, error_type='mse'):

        def mean_relative_error(orig, est):
            orig, est = np.array(orig), np.array(est)
            epsilon = 1e-8
            mre = np.mean(np.absolute(orig - est) / (orig + epsilon))
            return mre
        
        def mean_squared_error(X, Y):
            X, Y = np.array(X), np.array(Y)
            mse = np.mean((X-Y) ** 2)
            return mse

        def two_2_one(indices, shape):
            """
            Converts 2D indices to 1D indices using np.ravel_multi_index
            """
            return np.ravel_multi_index((indices[:, 0], indices[:, 1]), shape)

        def two_2_one_column(shape, j):
            """
            Convert the indices of the j-th column of a 2D tensor X to their corresponding
            flattened indices.
            
            Returns:
            torch.Tensor: The 1D tensor of flattened indices.
            """
            M, N = shape
            if j >= N:
                raise ValueError(f"Column index j={j} is out of bounds for a tensor with {N} columns.")
            
            # Get the indices of the j-th column
            row_indices = torch.arange(M)
            col_indices = torch.full((M,), j)
            
            # Convert 2D indices to 1D indices
            indices = torch.stack((row_indices, col_indices), dim=1)
            flattened_indices = np.ravel_multi_index(indices.T.numpy(), (M, N))
            
            return torch.tensor(flattened_indices)
    
        def one_2_two(index, shape):
            """
            Convert a flat index to a pair of indices (row, col) using np.unravel_index.
            Ensures that all indices are integers before conversion.
            """
            if isinstance(index, list):
                index = torch.tensor(index)

            # Check if all elements are integers and convert to int if necessary
            if not torch.all(index.float() == index.int()):
                print(index)
                raise ValueError("All elements in the index must be integers.")

            index = index.int()  # Ensure the tensor is of integer type
            row_indices, col_indices = np.unravel_index(index.numpy(), shape)
            return torch.stack((torch.tensor(row_indices), torch.tensor(col_indices)), dim=1)

        Pis = np.array(self.Pis)
        M = self.ds.M
        N = self.ds.N

        tmp = self.return_counts_as_anndata()

        X_hat = tmp[0].X.flatten().tolist().copy()
        X_theta = tmp[0].layers["X_hat_theta"].flatten().tolist().copy()
        if self.configs_inf['model_likelihood'] == 'ZINB':
            X_pi = tmp[0].layers["X_hat_pi"].flatten().tolist().copy()
        X_omega = tmp[0].layers["px_omega"].flatten().tolist().copy()
        generative = self.ds.anndatas_orig[0].layers['truth'].flatten().tolist().copy()

        masked_indices = two_2_one(self.ds.mask_indices, (M,N)).tolist()
        masked_indices_array = np.array(masked_indices)


        print("Imputation Experiment: Choosing random unmasked to match masked entries.")
        unmasked_indices, unmasked_indices_2d = [], []
        for j in tqdm(range(N)):
            feasible = two_2_one_column((M,N), j)

            possible_column_indices = [idx for idx in feasible if idx not in masked_indices_array]
            num_indices_to_zero = int(Pis[j] * M)
            
            rtmp = random.sample(possible_column_indices, num_indices_to_zero)
            unmasked_indices.extend(rtmp)
            rrtmp = one_2_two(rtmp, (M,N))
            unmasked_indices_2d.extend(rrtmp)


        # Converting lists to numpy arrays if needed later
        unmasked_indices = np.array(unmasked_indices)
        unmasked_indices_2d = np.array(unmasked_indices_2d)

        total_size = self.ds.mask_indices.shape[0] + len(unmasked_indices_2d)
        transcripts = np.empty(total_size, dtype=object)
        patients = np.empty(total_size, dtype=object)
        pis = np.empty(total_size, dtype=object)
        is_masked = np.empty(total_size, dtype=bool)
        mu_data = np.empty(total_size, dtype=object)
        X_hat_data = np.empty(total_size, dtype=object)
        theta_hat_data = np.empty(total_size, dtype=object)
        X_hat_pi = np.empty(total_size, dtype=object)
        X_hat_omega = np.empty(total_size, dtype=object)

        print("Imputation Experiment: Collecting masked entries.")
        for idx in tqdm(range(self.ds.mask_indices.shape[0])):
            current_2d = self.ds.mask_indices[idx]
            current_1d = masked_indices[idx]
            transcript = current_2d[1]
            patient = current_2d[0]

            pis[idx] = Pis[transcript]
            transcripts[idx] = transcript.item()
            patients[idx] = patient.item()
            is_masked[idx] = True
            mu_data[idx] = generative[current_1d]
            X_hat_data[idx] = X_hat[current_1d]
            theta_hat_data[idx] = X_theta[current_1d]
            if self.configs_inf['model_likelihood'] == 'ZINB':
                X_hat_pi[idx] = X_pi[current_1d]
            X_hat_omega[idx] = X_omega[current_1d]
            

        print("Imputation Experiment:Collecting unmasked entries.")
        offset = self.ds.mask_indices.shape[0]
        for idx in tqdm(range(len(unmasked_indices_2d))):
            current_2d = unmasked_indices_2d[idx]
            current_1d = unmasked_indices[idx]
            transcript = current_2d[1]
            patient = current_2d[0]
            
            transcripts[offset + idx] = transcript.item()
            patients[offset + idx] = patient.item()

            pis[offset + idx] = Pis[transcript]
            is_masked[offset + idx] = False
            mu_data[offset + idx] = generative[current_1d]
            X_hat_data[offset + idx] = X_hat[current_1d]
            theta_hat_data[offset + idx] = X_theta[current_1d]
            if self.configs_inf['model_likelihood'] == 'ZINB':
                X_hat_pi[offset +idx] = X_pi[current_1d]
            X_hat_omega[offset + idx] = X_omega[current_1d]

        if self.configs_inf['model_likelihood'] == 'NB':
            X_hat_pi = None

        df = pd.DataFrame({'patient_idx': patients,
                           'gene_idx': transcripts,
                           'pi_truth' : pis,
                           'pi_hat' : X_hat_pi,
                           'masked' : is_masked,
                           'truth_mu' : mu_data,
                           'mu_hat' : X_hat_data,
                           'dispersion_hat' : theta_hat_data,
                           'omega' : X_hat_omega
                           })
        df['pi_truth'] = df['pi_truth'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)

        df_masked = df[df['masked'] == True]
        df_unmasked = df[df['masked'] == False]

        if error_type=='mse':
            # Compute MSE for masked entries
            error_masked = mean_squared_error(df_masked['truth_mu'], df_masked['mu_hat'])
            # Compute MSE for unmasked entries
            error_unmasked = mean_squared_error(df_unmasked['truth_mu'], df_unmasked['mu_hat'])
        else:
            # compute MRE
            error_masked = mean_relative_error(df_masked['truth_mu'], df_masked['mu_hat'])
            error_unmasked = mean_relative_error(df_unmasked['truth_mu'], df_unmasked['mu_hat'])
            
        return error_masked, error_unmasked, df
     


    def return_counts_as_anndata(self):
        """
        Converts the raw and inferred gene expression data into an AnnData format for further analysis.

        :return: A list of AnnData objects, where each object represents a tissue or condition from the dataset.
        :rtype: List[anndata.AnnData]
        """
        adata = [self.ds.anndatas[i].copy() for i in range(self.ds.calT)]
        for i in range(len(adata)):
            adata[i].layers["original_counts"] = adata[i].X.copy()  

            # important variables to add
            if 'batch' in self.ds.anndatas[0].obs:
                adata[i].obs['batch'] = self.ds.anndatas[i].obs['batch'].copy()
            if 'subtype' in self.ds.anndatas[i].obs:
                adata[i].obs['subtype'] = self.ds.anndatas[i].obs['subtype'].copy()
            
        adata[0].X = self.output['X_hat_mu'][0].copy()
        adata[0].layers["X_hat_theta"] = self.output['X_hat_theta'][0].copy()
        if self.configs_inf['model_likelihood'] == 'ZINB':
            adata[0].layers["X_hat_pi"] = self.output['X_hat_pi'][0].copy()
        adata[0].layers["px_omega"] = self.output['px_omega'][0].copy()



        return adata
    
    def return_latent_space_as_anndata(self):
        """
        Creates an AnnData object from the latent variables (Z_L, Z_A, Z_Simple) depending on the model type.

        :return: A list of AnnData objects, where each object represents the latent space for a specific tissue or condition.
        :rtype: List[anndata.AnnData]

        :raises PreffectError: If the model type is neither 'simple', 'single', nor 'full', an error is raised indicating that the method is not implemented for the given model type.
        """

        anndata_list = []
        for idx in range(self.ds.calT):
            # Get the observation data for the current index
            # Create the AnnData object based on the configuration
            if self.configs_inf['type'] == 'simple':
                adata = ad.AnnData(X=np.copy(self.output['Z_simples'][idx]))
            elif self.configs_inf['type'] in ('single', 'full'):
                adata = ad.AnnData(X=self.output['Z_Xs'][idx])
            else:
                raise PreffectError('Not implemented for single and full models')
            
            adata.obs.index = self.ds.anndatas[idx].obs.index

            # important variables to add
            if 'batch' in self.ds.anndatas[idx].obs:
                adata.obs['batch'] = self.ds.anndatas[idx].obs['batch'].copy()
            if 'subtype' in self.ds.anndatas[idx].obs:
                adata.obs['subtype'] = self.ds.anndatas[idx].obs['subtype'].copy()
            if 'stroma_type' in self.ds.anndatas[idx].obs:
                adata.obs['stroma_type'] = self.ds.anndatas[idx].obs['stroma_type'].copy()

            # Append the AnnData object to the list
            anndata_list.append(adata)

        return anndata_list
    
    def save(self, results):
        """
        Saves the inference results to a file.

        :param results: The dictionary containing the inference results to be saved.
        :type results: Dict

        """
        fname = os.path.join(self.configs_inf['inference_path'], self.configs_inf['inference_key'])
        if self.configs_inf['adjust_vars']:
            fname = fname + '_' + str(self.configs_inf['adjust_to_batch_level']) + '.pkl'
        else:
            fname = fname + '_endo.pkl'
        with open(fname, 'wb') as file:
            pickle.dump(results, file, pickle.HIGHEST_PROTOCOL) 
        with open(fname + "_configs", 'wb') as file:
            torch.save(self.configs_inf, file, pickle.HIGHEST_PROTOCOL)

    def concatenate_2d_minibatch(self, L):
        """
        Concatenates sublists of tensors from multiple mini-batches into a single list of tensors.

        :param L: A list containing sublists, where each sublist consists of tensors from a particular mini-batch.
        :type L: List[List[torch.Tensor]]

        :return: A list of concatenated arrays. Each array in the list corresponds to a type of data across all
                mini-batches (e.g., all latent variable tensors concatenated into a single array).
        :rtype: List[numpy.ndarray]
        """
        tensor_arrays = [[tensor.cpu().numpy() for tensor in sublist] for sublist in L]
        transformed = [[tensor_arrays[j][i] for j in range(len(tensor_arrays))] for i in range(len(tensor_arrays[0]))]
        transformed = [np.concatenate(sublist, axis=0) for sublist in transformed]
        if self.configs_inf['number_padded'] != 0:
            transformed = [sublist[:-self.configs_inf['number_padded'], :] for sublist in transformed]
        return transformed

    def concatenate_1d_minibatch(self, L, full_idx):
        """
        Concatenates sublists of 1D tensors or lists from multiple mini-batches into a single list.

        :param L: A nested list where each inner list contains tensors or lists from a specific mini-batch.
        :type L: List[List[Union[torch.Tensor, List]]]
        :param full_idx: Indices representing the order or selection of elements to retain from the concatenated lists,
                        adjusted for any padding that might have been applied during batching.
        :type full_idx: List[int]

        :return: A list where each sublist corresponds to a concatenated and trimmed set of data across all mini-batches.
        :rtype: List[List]
        """
        minibatches = [ [tissue.cpu().squeeze().tolist() for tissue in minibatch]  for minibatch in L ]
        lib_size_factors = [[minibatches[j][i] for j in range(len(minibatches))] for i in range(len(minibatches[0]))]
        lib_size_factors = [sum(sublist, []) for sublist in lib_size_factors]
        l_prime = []
        for l in lib_size_factors:
            if self.configs_inf['number_padded'] != 0:
                l = l[:-self.configs_inf['number_padded']]
            l_prime.append( [l[i] for i in full_idx] )
        return l_prime


    # this also puts everything on the cpu and in numpy
    def reconstruct_from_minibatches(self, data, idx_batches):
        """
        Reconstructs full dataset arrays from minibatches, combining the various outputs
        from the model's minibatch processing into unified structures.

        :param data: List of dictionaries containing outputs from minibatches. Each dictionary should have keys
                 corresponding to different aspects of the model's output, such as 'latent_variables',
                 'lib_size_factors', 'X_hat', etc.
        :type data: List[Dict]
        :param idx_batches: List of tensors representing the indices of each minibatch within the overall dataset.
        :type idx_batches: List[List[torch.Tensor]]

        :return: A dictionary containing the reconstructed full dataset. The keys in this dictionary correspond to
                the unified arrays of outputs such as 'Z_Ls', 'lib_size_factors', 'X_hat_mu', 'X_hat_theta', 'Z_As',
                and potentially others depending on the model configuration and type.
        :rtype: Dict
        """
        full_data = {'Z_Ls' : None, 'lib_size_factors' : None, 'X_hat_mu' : None, 'X_hat_theta': None,
                     'X_hat_pi' : None, 'Z_simples': None, 'Z_As' : None, 'Z_Xs' : None, 'px_dispersion' : None}

        if self.configs_inf['type']=='simple': 
            idx_batches = [[torch.arange(self.ds.configs['M'])]]

            if 'Z_Ls' in data['latent_variables'] and data['latent_variables']['Z_Ls']:
                full_data['Z_Ls'] = [data['latent_variables']['Z_Ls'][0].cpu().numpy()]
                full_data['lib_size_factors'] = [data['lib_size_factors'][0].cpu().numpy()]

            full_data['px_omega'] = [data['px_omega'].cpu().numpy()]
            full_data['Z_simples'] = [data['latent_variables']['Z_simples'][0].cpu().numpy()]
            full_data['X_hat_mu'] = [data['X_hat'].mu.cpu().numpy()]
            full_data['X_hat_theta'] = [data['X_hat'].theta.cpu().numpy()]

            if self.configs_inf['model_likelihood'] == 'ZINB':
                full_data['X_hat_pi'] = [data['X_hat'].zi_probs().cpu().numpy()]


        elif self.configs_inf['type'] in ('single', 'full'):
            full_idx_numpy = [[t.cpu().numpy() for t in tt] for tt in idx_batches]  

            full_idx = [np.concatenate(t) for t in full_idx_numpy]  
            full_idx = [item for sublist in full_idx for item in sublist]

            if self.ds.configs['number_padded'] != 0:
                ### NOT TESTED ###
                full_idx = full_idx[:-self.ds.configs['number_padded']]

            if 'Z_Ls' in data[0]['latent_variables'] and data[0]['latent_variables']['Z_Ls'] != []:
                ### NOT TESTED ###
                full_data['Z_Ls'] = [lst[full_idx] for lst in self.concatenate_2d_minibatch([data[i]['latent_variables']['Z_Ls'] for i in range(len(data))])]
                minibatches = [data[i]['lib_size_factors'] for i in range(len(data))]
                full_data['lib_size_factors'] = self.concatenate_1d_minibatch(minibatches, full_idx)
                    
            full_data['Z_As'] = [lst[full_idx] for lst in self.concatenate_2d_minibatch([data[i]['latent_variables']['Z_As'] for i in range(len(data))])]
            full_data['Z_Xs'] = [lst[full_idx] for lst in self.concatenate_2d_minibatch([data[i]['latent_variables']['Z_Xs'] for i in range(len(data))])]

            full_data['DAs'] = [data[0]['DAs']]
            full_data['px_omega'] = [data[0]['px_omega'].cpu().numpy()]
            full_data['X_hat_mu'] = [data[0]['X_hat'].mu.cpu().numpy()]
            full_data['X_hat_theta'] = [data[0]['X_hat'].theta.cpu().numpy()]
            if self.configs_inf['model_likelihood'] == 'ZINB':
                full_data['X_hat_pi'] = [data[0]['X_hat'].zi_probs().cpu().numpy()]

        return full_data

        
    def visualize_lib_size(self, infer_obj):
        """
        Generates histograms comparing expected and inferred library sizes from the given Preffect object.

        :param infer_obj: An instance of the Inference class containing necessary data and configurations.
                      This object should have access to training datasets for original expressions and
                      inference results to fetch inferred library sizes.
        :type infer_obj: Inference

        :return: A matplotlib figure containing the histogram plots for both the observed and inferred
                library sizes, and optionally a scatter plot comparing them.
        :rtype: matplotlib.figure.Figure
        """
        pref_obj = infer_obj.parent

        def _generate_library_size_histogram(samples_np, position, color, title):
            """
            Generates a histogram of library sizes displayed in both log and normal scales.

            Args:
                samples_np (numpy.ndarray): Array of sample library sizes.
                position (int): The subplot position index in a matplotlib figure.
                color (str): Color specification for the histogram bars.
                title (str): Base title for the histogram. Titles for log and normal scale plots will be derived from this.

            Returns:
                None: This function modifies the matplotlib figure in place and does not return any values.
            """  
            plt.subplot(position)
            plt.hist(np.log(samples_np), bins=100, color=color)
            plt.xlabel("Observed (log) library sizes")
            plt.ylabel("Count")

            title_log = title + ": log(library)"
            plt.title(title_log)
            plt.text(-0.1, 1.1, title, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, rotation='horizontal', fontweight='bold', color=color)

            plt.subplot(position + 1)
            plt.hist(samples_np, bins=100, color=color)
            plt.xlabel("Sample library size")
            plt.ylabel("Count")
            title_norm = title + " library size"
            plt.title(title_norm)
        
        # compute observed library size and change to numpy
        real_lib_size = torch.sum(infer_obj.ds.Xs[0].cpu(), dim=1).squeeze().detach().cpu().numpy()

        # Generate histograms for two datasets
        plt.figure(figsize=(12, 18))
        _generate_library_size_histogram(real_lib_size, 321, "blue", title="Observed")
        
        # get inferred library sizes (only if infer_lib_size==True)
        if pref_obj.configs['infer_lib_size']:
            inferred_library = infer_obj.output['lib_size_factors'][0].squeeze()
            _generate_library_size_histogram(inferred_library, 323, "green", "Fit")
    
            ax_scatter = plt.subplot(313) 
            ax_scatter.scatter(real_lib_size, inferred_library)
            ax_scatter.set_xlabel('Observed library size')
            ax_scatter.set_ylabel('Fitted library size')
            # Add diagonal line
            min_val = min(min(real_lib_size), min(inferred_library))
            max_val = max(max(real_lib_size), max(inferred_library))
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line
        return plt


    # displays first 50 genes; if we have better ideas, I'm happy to hear it
    def visualize_gene_scatterplot(self, infer_obj):
        """
        Generates scatter plots comparing expected and fitted expression of the first 50 genes of the endogenous set.

        :param infer_obj: An instance of the Inference class containing necessary data and configurations.
                      This object should have access to training datasets for original expressions and
                      inference results to fetch inferred library sizes.
        :type infer_obj: Inference

        :return: A matplotlib figure containing the scatter plots for both the expected and inferred
                read counts of the first 50 genes in the dataset.
        :rtype: matplotlib.figure.Figure
        """
        pref_obj = infer_obj.parent
        #observed_counts = pref_obj.train_dataset.Rs[0]
        observed_counts = infer_obj.ds.Xs[0].cpu()

        # Create a 10x5 grid of subplots
        fig, axes = plt.subplots(10, 5, figsize=(15, 30))

        # Flatten the 2D array of axes for easy iteration
        axes = axes.flatten()

        # Loop to create 50 scatter plots
        for x in range(0, 50):
            observed_gene = observed_counts[:, x]  # choose a gene here
            inferred_gene = infer_obj.output["X_hat_mu"][0][:, x]

            # I need to add Pi to inferred_gene if ZINB
            if pref_obj.configs['model_likelihood'] == "ZINB":
                inferred_pi = infer_obj.output["X_hat_pi"][0][:, x]  # assuming data2 has the same structure for simplicity


            ax = axes[(x)]  # Get the current axis
            ax.scatter(observed_gene, inferred_gene)
            
            # Add diagonal line
            min_val = min(observed_gene.min(), inferred_gene.min())
            max_val = max(observed_gene.max(), inferred_gene.max())

            # adding a rug to the scatter plot
            sns.rugplot(x=observed_gene, ax=ax, color='blue')
            sns.rugplot(y=inferred_gene, ax=ax, color='blue')
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line

            # new addition, add Pi to X Label
            zero_mask = torch.eq(observed_gene, 0)

            num_zeros = torch.sum(zero_mask).item()
            pi = num_zeros / observed_gene.numel()

            # Adding titles and labels to each subplot
            ax.set_title('Scatter Plot: ' + infer_obj.ds.gene_names[x])
            ax.set_xlabel('Observed Counts (Xs) [pi=' + str(pi) + "]")

            ax.set_ylabel('Fitted Counts (Mu)')
            

        plt.tight_layout()
        return plt

    def visualize_batch_adjustment(self, infer_obj):
        """
        Generates side-by-side scatter plots comparing expected and fitted expression averages (per gene) between sample batches.
        The purpose is to see if batch correction was effective during training. 
        If >2 batches are provided, then all batches are compared to each other (e.g. batches 0+1, 0+2, and 1+2). 
        Since these comparisons will increase quadraticly, we limit comparisons to the first 5 batches.

        :param infer_obj: An instance of the Inference class containing necessary data and configurations.
                      This object should have access to training datasets for original expressions and
                      inference results to fetch inferred library sizes.
        :type infer_obj: Inference

        :return: A matplotlib figure containing the scatter plots, or None if there is only one batch or if
                batch correction is not being performed.
        :rtype: Optional[matplotlib.figure.Figure]
        """
        
        pref_obj = infer_obj.parent

        observed_counts = infer_obj.ds.Xs[0].cpu()
        inference_counts = torch.tensor(infer_obj.output["X_hat_mu"][0])

        # Apply Pi parameter
        if pref_obj.configs['model_likelihood'] == "ZINB":
            inferred_pi = torch.tensor(infer_obj.output["X_hat_pi"][0])

        # True if 'batch' is being corrected for
        is_batch_present = any(var[0] == 'batch' for var in pref_obj.configs['vars_to_correct'])

        # only run if batch is being corrected for
        if pref_obj.configs['correct_vars'] and is_batch_present:
            # get batch information 
            observed_corr = infer_obj.ds.Ks[0][0].cpu()

            # first column is always batch
            if observed_corr.ndim == 1:
                observed_batches = observed_corr
            else:
                observed_batches = observed_corr[:, 0]

            # get averages of the observed counts
            num_categories = len(np.unique(observed_batches))         
            subsets = [observed_counts[observed_batches == i] for i in range(num_categories)]
            averages = [subset.mean(dim=0) for subset in subsets]

            # and lets do the same for the fitted counts
            num_categories_inference = len(np.unique(observed_batches)) # this should be the same
            subsets_inference = [inference_counts[observed_batches == i] for i in range(num_categories_inference)]

            averages_inference = [subset.mean(dim=0) for subset in subsets_inference]

            num_plots = num_categories * (num_categories - 1) // 2 # floor division

            # if num_plots == 0, then there must be only 1 batch and thus no point to draw this plot
            if num_plots == 0:
                return None

            fig, axs = plt.subplots(num_plots, 2, figsize=(14, 7 * num_plots))

            # Plot scatter plots for each pair of categories
            plot_index = 0
            for i in range(num_categories):
                for j in range(i + 1, num_categories):

                    # comparisons increase factorially (3 batches, 3! comparisons); set limit to avoid too many comparisons
                    if i > 5 or j > 5:
                        continue

                    # Observed scatter plot
                    ax = axs[plot_index, 0] if num_plots > 1 else axs[0]
                    ax.scatter(averages[i], averages[j])
                    min_val = min(averages[i].min(), averages[j].min())
                    max_val = max(averages[i].max(), averages[j].max())

                    # rug plot
                    sns.rugplot(x=averages[i], ax=ax, color='blue')
                    sns.rugplot(y=averages[j], ax=ax, color='blue')

                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line
                    ax.set_title(f'Observed Average: Batch {i} vs Batch {j}')
                    ax.set_xlabel(f'Observed Average Counts (Batch {i})')
                    ax.set_ylabel(f'Observed Average Counts (Batch {j})')

                    # And now the inferred values
                    ax = axs[plot_index, 1] if num_plots > 1 else axs[1]
                    ax.scatter(averages_inference[i], averages_inference[j])
                    min_val = min(averages_inference[i].min(), averages_inference[j].min())
                    max_val = max(averages_inference[i].max(), averages_inference[j].max())

                    sns.rugplot(x=averages_inference[i], ax=ax, color='blue')
                    sns.rugplot(y=averages_inference[j], ax=ax, color='blue')

                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line
                    ax.set_title(f'Fitted Average: Batch {i} vs Batch {j}')
                    ax.set_xlabel(f'Fitted Average Counts (Batch {i})')
                    ax.set_ylabel(f'Fitted Average Counts (Batch {j})')

                    plot_index += 1

            plt.tight_layout()
            return plt


    def visualize_fraction_pi_per_gene(self, infer_obj):
        """
        Generates scatter plots comparing expected fraction of zeroes and the mean Pi parameter per gene.
        This is to compare the number of zeroes present before and after going through a ZINB model.

        Args:
            pref_obj (Preffect): An instance of the Preffect class containing necessary data and configurations.
                This object should have access to training datasets for original expressions and inference results
                to fetch inferred library sizes.

        Returns:
            matplotlib.pyplot: A matplotlib figure containing a scatter plot.
        """
        pref_obj = infer_obj.parent
        #original_expression = pref_obj.train_dataset.Rs[0]
        original_expression = infer_obj.ds.Xs[0].cpu()

        # True if 'batch' is being corrected for in `vars_to_correct`
        is_batch_present = any(var[0] == 'batch' for var in pref_obj.configs['vars_to_correct'])

        # only ZINB will have the pi parameter
        if pref_obj.configs['model_likelihood'] == "ZINB":
            inferred_pi = infer_obj.output["X_hat_pi"][0]  # assuming data2 has the same structure for simplicity
                
            observed_corr = infer_obj.ds.Ks[0][0].cpu()

            if observed_corr.ndim == 1:
                observed_batches = observed_corr
            else:
                observed_batches = observed_corr[:, 0]

            # get averages of the observed counts
            num_categories = len(np.unique(observed_batches))         
            subsets = [original_expression[observed_batches == i] for i in range(num_categories)]

            # Compute the fraction of zeroes for each gene in each batch
            fraction_zeroes_Xs = [(subset == 0).float().mean(axis=0) for subset in subsets]

            # and lets do the same for the fitted counts
            num_categories_inference = len(np.unique(observed_batches)) # this should be the same
            subsets_inference = [inferred_pi[observed_batches == i] for i in range(num_categories_inference)]
            prob_zeroes_inference = [subset.mean(axis=0) for subset in subsets_inference]

            # handles if there's only 1 batch
            if num_categories_inference == 1:
                fig, ax = plt.subplots(figsize=(7, 7))
                axs = [ax]  # Convert the single Axes object to a list for consistency
            else:
                fig, axs = plt.subplots(num_categories_inference, 1, figsize=(7, 7 * num_categories_inference))

            # Plot scatter plots for each pair of categories
            for i in range(num_categories):

                axs[i].scatter(fraction_zeroes_Xs[i], prob_zeroes_inference[i])
                    
                # Add diagonal line
                min_val = min(fraction_zeroes_Xs[i].min(), prob_zeroes_inference[i].min())
                max_val = max(fraction_zeroes_Xs[i].max(), prob_zeroes_inference[i].max())

                # lets add a rug as well
                sns.rugplot(x=fraction_zeroes_Xs[i], ax=axs[i], color='blue')
                sns.rugplot(y=prob_zeroes_inference[i], ax=axs[i], color='blue')

                axs[i].plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line

                # Adding titles and labels to each subplot
                axs[i].set_title(f'Fraction of Zeroes vs /hat Pi: Batch {i}')
                axs[i].set_xlabel('Fraction of Zeroes per Gene')
                axs[i].set_ylabel('Mean \hat Pi per Gene')
                axs[i].set_xlim(left=-0.01)
                axs[i].set_ylim(bottom=-0.01)
        
            return plt

    

    def visualize_libsize_and_dispersion(self, infer_obj):
        """
        Generates a series plots to compare gene means, dispersion and library size.

        :param infer_obj: An instance of the Inference class containing necessary data and configurations.
                      This object should have access to training datasets for original expressions and
                      inference results to fetch inferred library sizes [[3]][[6]][[8]].
        :type infer_obj: Inference

        :return: A matplotlib figure containing the 4-plot visualization.
        :rtype: matplotlib.figure.Figure
        """
        pref_obj = infer_obj.parent

        def _plot_libsize(pref_obj, infer_obj, ax=None, title="Library size"):
            """
            """
            #X = np.array(pref_obj.train_dataset.Rs[0])
            X = np.array(infer_obj.ds.Xs[0].cpu())
            
            if pref_obj.configs['infer_lib_size']:
                estimated_library_size = infer_obj.output["lib_size_factors"]
            else:
                estimated_library_size = np.sum(X, 1)
            actual_library_size = np.sum(X, 1)

            if ax is None:
                fig, ax = plt.subplots()
            ax.scatter(
                np.log10(actual_library_size), np.log10(estimated_library_size),
                color="black", s=2, alpha=0.5, label="cell", marker=".",
            )
            ax.set_xlabel("Observed log10(UMI)")
            ax.set_ylabel("Estimated log10(libsize)")
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x, color="red", linestyle="dashed", label="x=y")
            ax.legend(scatterpoints=3, frameon=False)
            ax.set_title(title)

        def _plot_mean_dispersion2(pref_obj, infer_obj, ax=None, title="Dispersion vs mean (gene)", logxy=True):
            """
            """            
            is_batch_present = any(var[0] == 'batch' for var in pref_obj.configs['vars_to_correct'])
            #X = np.array(pref_obj.train_dataset.Rs[0])
            X = np.array(infer_obj.ds.Xs[0].cpu())


            px_rate_obs = X.mean(0)

            px_rate_generated = infer_obj.output["X_hat_mu"][0].mean(0)
            px_r = infer_obj.output["X_hat_theta"][0][0, :]
            if ax is None:
                fig, ax = plt.subplots()

            ax.scatter(
                px_rate_generated, px_r, label="generated", color="black", s=5, alpha=0.5
            )
            ax.set_xlabel("NB mean")
            ax.set_ylabel("NB dispersion")
            ax.set_title(title)
            # ax.legend(frameon=False)
            if logxy:
                ax.set_yscale("log")
                ax.set_xscale("log")


        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(221)
        _plot_libsize(pref_obj, infer_obj, ax=ax, title="Library size")
        ax = fig.add_subplot(222)
        _plot_mean_dispersion2(pref_obj, infer_obj, ax=ax)
        fig.tight_layout()
        
        px_rate_generated = infer_obj.output["X_hat_mu"][0].mean(0)
        dispersions = infer_obj.output["X_hat_theta"][0][0, :]

        ax = fig.add_subplot(223)
        ax.hist(dispersions, color="black")
        ax.set_xlabel("NB Dispersion")
        ax.set_title("NB Disperion")
        ax.set_ylabel("Count")
    
        ax = fig.add_subplot(224)
        ax.hist(px_rate_generated, color="black")
        ax.set_xlabel("NB mean")
        ax.set_ylabel("Count")
        ax.set_title("NB mean")
        fig.tight_layout()
        return plt


    def visualize_latent_recons_umap(self, infer_obj, my_cmap=None):
        """
        UMAP of latent space; assumes 'batch' is in obs table of AnnData
        """
        pref_obj = infer_obj.parent

        def _plot_latent_umap(pref_obj, infer_obj, my_cmap, ax=None, title="Latent"):
            """
            """
            if pref_obj.configs['type'] == "simple":
                latent = infer_obj.output['Z_simples'][0]
            elif pref_obj.configs['type'] == "single":
                latent = infer_obj.output['Z_As'][0]
            else:
                # the first M samples should be the tumour set. Need to confirm.
                latent = infer_obj.output['Z_As'][0][:infer_obj.ds.M]

            reducer = umap.UMAP(
                metric="cosine", n_neighbors=30
            ) 


            embedding = reducer.fit_transform(latent)
            embedding_df = pd.DataFrame(embedding, columns=["0", "1"])

            inference_corr = infer_obj.ds.Ks[0][0].cpu()

            if inference_corr.ndim == 1:
                inference_batch = inference_corr
            else:
                inference_batch = inference_corr[:, 0]


            if ax is None:
                fig, ax = plt.subplots()
            scatter = ax.scatter(
                embedding_df["0"].values,
                embedding_df["1"].values,
                c=inference_batch,
                s=3,
                cmap=my_cmap,
            )
            ax.set_title("Latent space")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.grid(False)


        def _plot_recons_error_umap(pref_obj, infer_obj, my_cmap, ax=None, title="Reconstruction error"):
            """
            """
            X = infer_obj.ds.Xs[0].cpu()

            px_rate_generated = infer_obj.output["X_hat_mu"][0]
            px_r = infer_obj.output["X_hat_theta"][0]
            
            X = Tensor(X)
            px_rate = Tensor(px_rate_generated)
            px_r = Tensor(px_r)

            # get batch
            inference_corr = infer_obj.ds.Ks[0][0].cpu()
            if inference_corr.ndim == 1:
                inference_batch = inference_corr
            else:
                inference_batch = inference_corr[:, 0]

            gene_cell_loss = (
                NegativeBinomial(mu=px_rate, theta=px_r).log_prob(X).detach().cpu().numpy()
            )
            
            # this is a check for safety
            if np.isnan(gene_cell_loss).any() or np.isinf(gene_cell_loss).any():
                raise ValueError("Input contains NaN or Inf values")

            reducer = umap.UMAP(metric="cosine", n_neighbors=30, random_state=42, n_jobs=1, verbose=False, low_memory=True)

            try:
                embedding = reducer.fit_transform(gene_cell_loss)
                # Monitor memory usage after UMAP

            except Exception as e:
                print(f"An error occurred: {e}")
                return None



            embedding_df = pd.DataFrame(embedding, columns=["0", "1"])

            if ax is None:
                fig, ax = plt.subplots()
            scatter = ax.scatter(
                embedding_df["0"].values,
                embedding_df["1"].values,
                c=inference_batch,
                s=3,
                cmap=my_cmap,
            )
            ax.minorticks_off()
            ax.set_title(title)
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.grid(False)

        # user can set plot colours; if not, we use these as default
        if my_cmap is None:
            simple_colors = (
                sns.color_palette("muted").as_hex() + sns.color_palette("Set3").as_hex()[4:6]
            )
            my_cmap = ListedColormap(simple_colors)

        # main section
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(121)
        _plot_recons_error_umap(pref_obj, infer_obj, my_cmap=my_cmap, ax=ax, title="Reconstruction error")
        ax = fig.add_subplot(122)
        _plot_latent_umap(pref_obj, infer_obj, my_cmap=my_cmap, ax=ax, title="Reconstruction error")
        fig.tight_layout()
        return fig


    def save_visualization(self, vlib, filename=None):
        """
        Saves a matplotlib figure to a specified directory as a PDF file.

        :param vlib: The matplotlib figure to be saved.
        :type vlib: matplotlib.figure.Figure
        :param filename: The name of the file to save the figure as, including the file extension.
        :type filename: str

        :raises PreffectError: If `filename` is not provided.
        """
        if filename is None:
            raise PreffectError('Must specify a filename to store a visualization.')
        
        # Save the figure
        if vlib is not None:
            vlib.savefig(filename)
            if not isinstance(vlib, plt.Figure):
                vlib.close()
            
        

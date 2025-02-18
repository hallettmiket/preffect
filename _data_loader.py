import os
import numpy as np
import torch
import torch.sparse
import torch.nn.functional as F
import pandas as pd
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
import fnmatch
import re
import random
import warnings
import logging
import anndata as ad
import itertools
from _utils import (
    logging_tensor, 
    categorical_correction_variable, 
    adjusted_categorical_correction_variable, 
    check_int_and_uniq, 
    calculate_and_sort_by_iqr,
    _indices_to_tensor  )

input_log = logging.getLogger('input')

class FFPE_dataset(Dataset):
    """
    FFPE_dataset: start of data loader
    """

    def __init__(self, configs, learning_type, parent): 

        self.configs = configs
        self.learning_type = learning_type
        self.parent = parent

        self.anndatas_orig = self.load_anndata()
        # self.N, M, calT, .... are defined below primarily in from_anndata_2_numpy()
        # self.ks, levels of korrection variables; first column is always batch if available.
        # self.Ks, korrection variables; first is batch if available.
        testing = self.ground_truth()
        # I have no idea why, but 'truth' just would not exist if 'else' conditional in ground_truth() was used
        # so I resorted to this
        if 'truth' not in self.anndatas_orig[0].layers:
            for i in range(self.configs['calT']):
                self.anndatas_orig[i].layers['truth'] = testing[i]



        # print(self.anndatas_orig[0].layers['truth'][0:3][0:3])

        if not self.check_fidelity_of_anndata():
            raise ValueError('Something wrong with input data. Check log.')

        if self.configs['calT'] > 1:
            # only "full" models
            if (self.configs['type'] == 'full'):
                self.anndatas, self.sample_tissue_map = self.harmonize_samples(self.anndatas_orig)
            else:
                self.configs['calT'] = 1
                self.calT = 1
                self.anndatas = self.anndatas_orig
                input_log.info(f'Only full models allow for >1 tissue. Defaulting calT to 1.')
        else: 
            self.anndatas = self.anndatas_orig

        self.anndatas = self.select_genes(self.anndatas)
        self.anndatas = self.select_samples(self.anndatas)
        self.from_anndata_2_numpy(self.anndatas)
        self.setup_categorical_variables()
        self.from_numpy_2_tensors()

        self.possible_masking_strategies = {
            'MCAR': self.MCAR, 'MAR': self.MAR, 'MNAR': self.MNAR, 'NONE': None}

        if self.learning_type=='inference':
            self.configs['masking_strategy'] = None

        if self.configs['masking_strategy']==None:
            self.Xs = self.clamp_tensor_values(self.Xs)
            self.mask_indices = None
            if self.configs['adj_exist']:
                self.As_mask = self.As_orig.copy()
                self.As_ej_index_masked = self.As_ej_index.copy()
                self.As_mask_indices = None
            
        else:
            Xs, Xs_indices, As_masked, As_indices, As_ej_index_masked = self.possible_masking_strategies[
                self.configs['masking_strategy']](self.configs)
            self.Xs = self.clamp_tensor_values(Xs)
            self.mask_indices = Xs_indices
            if self.configs['adj_exist']:
                self.As_mask = As_masked
                self.As_mask_indices = As_indices
                self.As_ej_index_masked = As_ej_index_masked

        self.configs['batch_present'] = True
        for adata in self.anndatas_orig:
            if 'batch' not in adata.obs.columns:
                self.configs['batch_present'] = False
                break
        logging_tensor(input_log, self.Xs,
                       "\nFFPE_dataset init - Xs after dropout+clamping")
        log_string=f"sample names: ({str(len(self.sample_names))})\
                        {str(self.sample_names)}\n\
                        gene names: ({str(len(self.gene_names))})\
                        {str(self.gene_names)}\n\
                        N {str(self.N)} M {str(self.M)} calT {str(self.calT)}\
                        Sample names: ({str(len(self.sample_names))})\
                        {str(self.sample_names)}\
                        Gene names: ({str(len(self.gene_names))})\
                        {str(self.gene_names)}"
        if configs['adj_exist']:
            log_string = log_string + f"\nNumber ejs A matrices: {str([A.shape[1] for A in self.As_ej_index_masked])}"
        input_log.info(log_string)

    def load_anndata(self):
        """
        This loads anndata files (h5ad) from a given directory (specified in
        configs). All files that have the name structure
        *tau_x.h5ad where x is an integer are loaded x determines
        the order of tissues.

        Args:
            **anndata_dir:** location of all anndata files.
            anntarget_dataset: specifies a specific dataset within anndata_dir
            learning_type: species whether it is 'train', 'validation' or 'test'            

        Returns:
            One anndata datastructure for each valid file identified
            sorted by tau.
        """

        learning_path = self.learning_type + "/"
        if self.learning_type == "inference" or self.learning_type == "impute_experiment": 
            final_path = os.path.join(self.configs['input_inference_anndata_path'])
            ann_path =  os.listdir(final_path)
        else:
            final_path = os.path.join(self.configs['input_anndata_path'], learning_path)
            ann_path = os.listdir(final_path)

        pattern_annfiles = r'tau_(\d+)\.h5ad'
        ann_files = sorted(fnmatch.filter(ann_path, '*.h5ad'))
        
        if len(ann_files) > self.configs['calT']:
            ann_files = ann_files[:self.configs['calT']]
            self.configs['calT'] = len(ann_files)
            input_log.info(f"Warning: declared calT ({self.configs['calT']}) < # of AnnData files found ({len(ann_files)}).")
        elif len(ann_files) < self.configs['calT']:
            self.configs['calT'] = len(ann_files)
            input_log.info(f"Warning: declared calT ({self.configs['calT']}) > # of found AnnData files ({len(ann_files)}). Reseting calT to {len(ann_files)}")

        ann_files_list = [int(re.search(pattern_annfiles, s).group(1))
                          for s in ann_files if re.search(pattern_annfiles, s)]

        # sorted by the *tau_{int}.h5ad at end of file.
        # if the folder contains multiple files with the same tau (e.g. two tau_1s), die
        check_int_and_uniq(ann_files_list)
        
        anndatas = [ad.read_h5ad(os.path.join(final_path, ann_files[s-1]))
                    for s in ann_files_list]
        self.configs['adj_exist'] = True
        for i, adata in enumerate(anndatas):
            if not 'sample_sample_adj' in adata.obsm:
                input_log.info(f'Missing adjacency matrix for tissue {i}. Defaulting to simple model.')
                self.configs['adj_exist'] = False
                self.configs['type']='simple'
                anndatas = [anndatas[0]] # just keep first


        in_log = f"Loading dataset: {self.configs['input_anndata_path']} of type: {self.learning_type}\
                \nFinal calT: {str(self.configs['calT'])}\
                \nFound anndata files: {str(ann_files)}\
                \nDoes Adjacency files exist?\
                {str(self.configs['adj_exist'])}\
                \nFinal model: {str(self.configs['type'])}"
        input_log.info(in_log)

        return anndatas

    def check_fidelity_of_anndata(self):
        """
        This function checks each input AnnData object for the following:

        1. A valid expression matrix (AnnData 'X' matrix).

        2. Gene IDs (AnnData 'var' must exist with column 'gene').

        3. Sample IDs (AnnData 'obs' table must exist; expects 'obs_names' to be sample IDs).

        4. Presence of an Aj correlation matrix. If not found, the `configs` is modified so that adjacency information is not considered in downstream computations.

        5. A "sample_association" must exist in the `obsm` of each AnnData, indicating which samples are associated with each other (tau_1, tau_2, etc.). Consequently, the sample names between the AnnData tables (N=calT) are expected to be unique. If a sample is not associated with a certain tissue, the cell for that unassociated tissue should read NA or NaN.

        Args:
            **adatas (List[AnnData]):** A list of original AnnData objects loaded from files.

        Returns:
            **bool:** True if all checks pass.

        Raises:
            None
        """

        try:
            # if calT > number of AnnData tables found
            if len(self.anndatas_orig) < self.configs['calT']:
                raise ValueError(f"Too few AnnData objects relative to declared calT")
        except ValueError as e:
            input_log.info(f"Too few AnnData objects relative to declared calT")

        for index, adata in enumerate(self.anndatas_orig):
            try:
                if adata.X is None:
                    raise ValueError(f"Matrix 'X' (expression) in AnnData object is empty for tissue {index}.")
            except ValueError as e:
                input_log.info(f"Matrix 'X' (expression) in AnnData object is empty for tissue {index}.")

            try:
                if adata.var.empty:
                    raise ValueError(f"Var (for gene IDs) is missing  in AnnData object for tissue {index}.")
            except ValueError as e:
                input_log.info(f"Var (for gene IDs) is missing  in AnnData object for tissue {index}.")

            try:
                if 'gene' not in adata.var.keys():
                    raise ValueError(f"Gene IDs are missing  in AnnData object for tissue {index}.")
            except ValueError as e:
                input_log.info(f"Gene IDs are missing  in AnnData object for tissue {index}.")

            try:
                if adata.obs.empty:
                    raise ValueError(f"obs  in AnnData object for tissue {index}.")
            except ValueError as e:
                input_log.info(f"obs  in AnnData object for tissue {index}.")

            try:
                if self.configs['calT'] > 1 and 'sample_association' not in adata.obsm.keys():
                    raise ValueError(f"obsm in AnnData is missing associations_table for tissue {index}.")
            except ValueError as e:
                input_log.info(f"obsm in AnnData is missing associations_table for tissue {index}.")

        return True


    def harmonize_samples(self, anndata):
        """
        This function reads "sample_association" in the obsm, and uses it to order associated samples to the same rows in X.

        - "sample_association" is a table of calT cols ("tau_1", etc) and indicates which tissues are related.
        - "sample_association" combined and used to order the table X, obs, and obsm table "sample_sample_adj" (var/varm unaffected).
        - If a sample doesn't have an associated sample in one or more tissues, the other cols for those tissues in this table should be 'NaN'.
        - We expect AnnData 'obs_names' should be unique across all AnnData tables provided.

        Returns:
            List of Anndata of length calT; restructured to order rows the same across associated samples; if no association, a "ghost" entry is added:
            
            - NA/NaN are converted to a unique name.
            - A new row with this name is added to 'obs'; 'NaN' added to each entry (for easy identification of ghost samples).
            - A new row with this name is added to 'X', where expression of each gene is zero.

            The AnnData tables X, obs, and the required obsm sparse matrix "sample_sample_adj" are then re-arranged to align associated samples to the same row. It is expected that each sample has a maximum of one associated sample per calT (e.g. a sample can only have one association to a sample in a different tissue).

        Raises:
            None due to filtering step in 'check_fidelity_of_anndata'.
        """
        # Step 1: Assoc. across all AnnData tables are combined; dups dropped
        sample_assoc = [anndata[i].obsm['sample_association']
                        for i in range(len(anndata))]
        # this collects all associations across tissues
        sample_assoc_nodup = pd.concat(sample_assoc).drop_duplicates().copy()
        sample_assoc_nodup.index.names = [None]
        sample_assoc_nodup.replace('NA', np.nan, inplace=True)

        for i in range(self.configs['calT']):
            tau_find = "tau_" + str(i + 1)


            # No. "ghost samples" po add to AnnData tables
            num_ghost_sample = sample_assoc_nodup[[tau_find]].isna().sum()[0]

            # we then want to add that many new samples to the AnnData table,
            # with NA for obs and 0 for expr. (unless ave_ghost_expr is true)
            new_sample_annotations = pd.DataFrame(
                data=np.nan,
                index=[tau_find + "_" + str(x + 1)
                       for x in range(num_ghost_sample)],
                columns=anndata[i].obs.columns
            )

            # we don't average by batch as ghosts don't have assoc. batch
            gene_average = np.nanmean(anndata[i].X, axis=0)
            # in stroma simulation, some genes were always "nan", hence warning
            gene_average = np.round(np.nan_to_num(gene_average)).astype(int)
            new_sample_expression_data = np.transpose(
                np.repeat(gene_average[:, np.newaxis],
                          num_ghost_sample, axis=1)).astype(np.float32)

            new_anndata = ad.AnnData(new_sample_expression_data,
                                     obs=new_sample_annotations,
                                     var=anndata[i].var)
            concat_datas = ad.concat([anndata[i], new_anndata],
                                     join='inner').copy()

            # lets modifiy 'sample_assoc_nodup' so "NaN" match new names
            unique_id_generator = (tau_find + "_" + str(x)
                                   for x in itertools.count(1))
            sample_assoc_nodup[tau_find] = sample_assoc_nodup[tau_find].apply(
                lambda x: next(unique_id_generator) if pd.isna(x) else x)

            # need to rearrange X and obs to match 'sample_assoc_nodup'
            index_map = {name: i for i,
                         name in enumerate(concat_datas.obs_names)}
            new_order = [index_map[name]
                         for name in sample_assoc_nodup[tau_find]]
            
            # after concat., var/varm/obsm are empty. 'adjacency' table
            # must be re-ordered to match new_order. "adjacency" is not
            # a required table, so arrangement will only occur when present
            if self.configs['adj_exist']:
                new_datas_sample_adj = anndata[i].obsm['sample_sample_adj'].tocoo()
                order_map = {old_idx: new_idx
                             for new_idx, old_idx in enumerate(new_order)}

                # Reorder the row indices
                new_rows = np.array([order_map[row_idx]
                                     for row_idx in new_datas_sample_adj.row])
                new_cols = np.array([order_map[col_idx]
                                     for col_idx in new_datas_sample_adj.col])

                # the sparse matrix shape will have to be changed, as
                # X/Y now should be the length of "sample_association"
                new_shape = (concat_datas.n_obs, concat_datas.n_obs)

                # Create the reordered COO matrix
                reordered_coo = coo_matrix(
                    (new_datas_sample_adj.data, (new_rows, new_cols)),
                    shape=new_shape
                )

                # Convert the reordered COO matrix to the desired sparse matrix
                # format; also re-arranges matrix to coordinate order
                ord_mtx = reordered_coo.tocsr()
                concat_datas.obsm['sample_sample_adj'] = ord_mtx.astype(
                    np.float32)

                # var and varm unaffected by addition of ghost samples
                concat_datas.varm = anndata[i].varm
                 
            
            # should be done regardless if adj_exists
            concat_datas.var = anndata[i].var

            # and the "truth" layer will need to be expanded for the ghost samples


            repeats_per_row = len(concat_datas.obs) // len(anndata[i].layers['truth'])
            extra_rows = len(concat_datas.obs) % len(anndata[i].layers['truth'])

            # Repeat each row accordingly
            truth_with_ghosts = np.repeat(anndata[i].layers['truth'], repeats_per_row, axis=0)

            # If extra rows are needed, add them from the beginning
            if extra_rows > 0:
                truth_with_ghosts = np.vstack([truth_with_ghosts, anndata[i].layers['truth'][:extra_rows, :]])

            concat_datas.layers['truth'] = truth_with_ghosts

            # re-arrangements written to anndata variable; this line was moved before re-arrangement for some reason
            anndata[i] = concat_datas[new_order]

        return anndata, sample_assoc_nodup

    def select_genes(self, anndatas):
        """
        Perform gene filtering steps, if desired. 

        - Limit to highest variable genes based on 'select_genes' parameter.
        
        - Remove highest expressed genes using trimming, if 'trim_high_expressed_genes' is true.
        """

        # If the user sets "select_genes" to a value < # genes in the AnnData, it will automatically reduce the dataset and adjust the gene/gene network
        self.configs['N'] = anndatas[0].n_vars
        if self.configs['select_genes'] < self.configs['N'] and self.configs['calT']==1:
            input_log.info("Reducing input genes to {self.configs['select_genes']} from {self.configs['N']}")
            sorted_indices = calculate_and_sort_by_iqr(expr=anndatas[0].X, 
                                                       gene_names = anndatas[0].var.gene, 
                                                       force_pam50 = self.configs['force_pam50'],
                                                       force_bc_genes= self.configs['force_bc_genes'])
            selected_genes_indices = sorted_indices[:self.configs['select_genes']] # limiting to genes with highest IQR
            
            anndatas = [anndatas[s-1][:, selected_genes_indices] for s in range(self.configs['calT'])]
            self.N = self.configs['select_genes']
        elif self.configs['select_genes'] > self.configs['N']:
            self.configs['select_genes'] = self.configs['N']


        # If the user activates high expression trimming
        if (self.configs['trim_high_expressed_genes']):
            if self.learning_type == "train":
                mean_expression = np.mean(anndatas[0].X, axis=0)

                # Determine the number of genes to trim
                trim_count = int(np.floor(self.configs['N'] * self.configs['trim_percentage']))

                #  Indices of genes to keep (all but the top 1% highest mean expressions)
                threshold_value = np.partition(mean_expression, -trim_count)[-trim_count]

                # Create a boolean mask for genes to keep (those below the threshold)
                keep_mask = mean_expression < threshold_value
                remove_mask = mean_expression >= threshold_value

                # print("t", anndatas[0][:, remove_mask].var['gene'])
                # Filter the AnnData object to keep only the desired genes
                anndatas = [anndatas[s-1][:, keep_mask] for s in range(self.configs['calT'])]
                
                self.trim_keep_mask = keep_mask
                # self.trim_remove_mask = remove_mask # for testing
            elif self.learning_type == "validation":
                # print("v", anndatas[0][:, self.parent.train_dataset.trim_remove_mask].var['gene'])
                anndatas = [anndatas[s-1][:, self.parent.train_dataset.trim_keep_mask] for s in range(self.configs['calT'])]
            else:
                # print("i", anndatas[0][:, self.parent.parent.train_dataset.trim_remove_mask].var['gene'])
                anndatas = [anndatas[s-1][:, self.parent.parent.train_dataset.trim_keep_mask] for s in range(self.configs['calT'])]


            self.configs['N'] = anndatas[0].n_vars

            # save the keep_mask and somehow apply to all other datasets


        return anndatas


    def select_samples(self, anndata):
        """
        Randomly selects samples from AnnDatas table based on the value set by configs['select_samples'].
        If configs['select_samples'] = float('inf') or <=0, all samples will be chosen normally.

        **Args**:
        
        - **anndata (list):** A list of anndata files (X sorted to match each other after harmonize_samples().

        **Returns**:
        
        - **anndatas (list):** Updated list of anndata files.

        """
        self.configs['M'] = anndata[0].n_obs

        # will not activate if performing inference
        if self.configs['select_samples']!=float('inf') and self.configs['task']=='train':
            assert self.configs['calT']==1, f"Currently can only select samples for one tissue scenarios."  # fix later.
            input_log.info("Sampling from the input samples: now {self.configs['select_samples']} from original {self.configs['M']}")

            # if requested samples > number of samples available, turn on replacement
            if self.configs['select_samples'] > self.configs['M']:
               self.configs['select_sample_replacement'] = True
               input_log.info("{self.configs['select_samples']} > {self.configs['M']}; replacement has automatically been turned on")


            # if requested samples < mini-batch size, then reduce the minibatch size
            if self.configs['select_samples'] < self.configs['mini_batch_size']:
               self.configs['mini_batch_size'] = int(self.configs['select_samples'])
               warnings.warn("self.configs[mini_batch_size] > self.configs[select_samples]; they are now both set to config[select_samples].")
          
            # expr_df = pd.DataFrame(anndata[0].X)
            # sorted_indices = expr_df.sample(n=self.configs['select_samples'], replace=self.configs['select_sample_replacement']).index
            #### sorted_indices = select_samples_from_table(orig_M=anndatas[0].X.shape[0])
            integers = np.arange(self.configs['M']).tolist()
            if self.configs['select_sample_replacement']:
                sampled_indices = random.choices(integers, k=int(self.configs['select_samples']))
            else:
                sampled_indices = random.sample(integers, k=int(self.configs['select_samples'])) 

            self.M = self.configs['select_samples']
            self.configs['M'] = self.configs['select_samples']
            # this should work for multiple tissues (must occur after harmonization)
            for i in range(self.configs['calT']):
               anndata[i] = anndata[i][sampled_indices, :]
              
               # adjust sample-sample matrix (the above line)
               if (self.configs['adj_exist']):
                   # the only way to avoid a warning when over-writing obsm is to copy the entire AnnData, and modify it.
                   actual_anndata = anndata[i].copy()
                   new_datas_sample_adj = actual_anndata.obsm['sample_sample_adj'].copy()
                   new_sparse_matrix = new_datas_sample_adj[:, sampled_indices]
                   actual_anndata.obsm['sample_sample_adj'] = new_sparse_matrix
                   
                   # Avoids a warning
                   anndata[i] = actual_anndata
                  
        return anndata

            
    
    def get_ghost_indices(self, tissue):
        """
        Retrieves indices of "ghost" samples in a particular tissue sample set.

        **Parameters**:
        
            **tissue** (int): An integer representing the index of the tissue in the `self.calT` range.

        **Returns**:
        
            **indices** (list[int]): List of indices in self.anndatas[tissue].obs.index.value that correspond to the ghost samples in `self.sample_tissue_map`.

        **Raises**:
        
            **AssertionError**: If the input `tissue` is not within the range defined by `self.calT`.
        """ 
        if (self.calT == 1):
            return None

        assert tissue in range(self.calT)
        
        indices = self.sample_tissue_map.iloc[:, tissue][
            self.sample_tissue_map.iloc[:, tissue].str.startswith(
                'tau_').tolist()]
        _from = list(indices.values)
        _into = list(self.anndatas[tissue].obs.index.values)
        return [_into.index(fr) for fr in _from]

    def from_anndata_2_numpy(self, adata):
        """
        Extracts info from anndata, creates several self vars for this dataset

        Args:
            **anndata_orig:** raw anndata structures read in from file. They have
            been checked for their fidelity (no information is missing)

        Returns:
            A series of self variables are created in the dataset object

        Raises:
            Not sure yet if there is anything to raise here.
        """
        # gene by sample
        self.Rs = [np.transpose(A.X.copy()).astype(float)
                   for A in adata]
        # convert all NaN's in expression to zero
        self.Rs = [np.nan_to_num(R.T) for R in self.Rs]
        self.M = self.Rs[0].shape[0]
        self.N = self.Rs[0].shape[1]

        self.calT = self.configs['calT']
        self.gene_names = adata[0].var.gene.tolist()
        self.sample_names = [A.obs_names.tolist() for A in adata]
        self.sample_metadata = [A.obs.copy() for A in adata]

        if self.configs['adj_exist']:
            self.As = [A.obsm['sample_sample_adj'].tocoo()
                   for A in adata]
        return None

    def setup_categorical_variables(self):
        """
        Configures the categorical variables for each tissue and sample in the dataset.

        **Uses**:
        
        - **self.configs['vars_to_correct']** (list[str]): List of column names in sample_metadata DataFrames that need to be one-hot encoded.
        
        - **self.configs['adjust_target_batch']** (list of tuples): List of tuples where the first element is the column name to adjust for and the second element is a target value for the adjustment. This is used during inference (only) to adjust the expression levels to a specific target batch.

        **Returns**:
        
        - **None**: This function modifies the instance variables in place.

        **Attributes Modified**:
        
        - **self.Ks**: List of one-hot encoded DataFrames for each sample's correction variables.
        
        - **self.ks**: List of integers representing the number of one-hot encoded columns for each sample's correction variables.

        **Raises**:
        
        - **AssertionError**: The input column names in 'correction_vars' should match with the column names in 'self.sample_metadata'.
        """
        if not self.configs['correct_vars'] or self.configs['vars_to_correct'] is None or len(self.configs['vars_to_correct']) == 0:
           self.Ks = [ [torch.zeros(self.M, dtype=torch.int32)] for i in range(self.calT)]
           
           self.ks = [[1] for i in range(self.calT)]

           self.cat_dict = dict()
           return

        # check that all the correction variables are present in the sample metadata
        correction_vars = {x[0] for x in self.configs['vars_to_correct']}
        sample_metadata = self.sample_metadata # should be a list of anndatas.
        
        # In turn, all the correction variables must in the list of metadata for that tissue.
        for i, metadata_df in enumerate(sample_metadata):
            assert set(correction_vars).issubset(set(metadata_df.columns)), (
                    f"correction variable(s) not in sample metadata of tissue {i}"
                )         
        
        Ks, ks, combined_cat_maps = [], [], []

        for i, metadata_df in enumerate(sample_metadata):

            tmp_k, tmp_levels, tmp_combined_cat_maps, batch_encoded_df, batch_cat_map = [], [], [], [], []
            for j, pr in enumerate(self.configs['vars_to_correct']):
                var, typ = pr[0], pr[1]
                

                if var=='batch' and self.configs['adjust_vars']==True:
                    if typ=='categorical':
                        batch_encoded_df, batch_cat_map, batch_levels = adjusted_categorical_correction_variable(metadata_df, var, self.configs['adjust_to_batch_level'])
                        # tmp_combined_cat_maps.append(cat_map)
                    else: # 'continuous':
                        batch_encoded_df = torch.full((metadata_df.shape[0], 1), self.configs['adjust_to_batch_level'], dtype=torch.float)
                        batch_levels = float('inf')
                        batch_cat_map = None
                else: # not the special case of adjusted batch variable
                    if typ=='categorical':
                        encoded_df, cat_map, lvls = categorical_correction_variable(metadata_df, var)
                        lvls = len(metadata_df[var].unique())
                        tmp_combined_cat_maps.append(cat_map)
                    else: # 'continuous
                        encoded_df = torch.tensor(metadata_df[var].values, dtype=torch.float)#.unsqueeze(1)
                        #lvls = float('inf')
                        lvls = None
                        tmp_combined_cat_maps.append(None)
                    tmp_k.append(encoded_df)
                    tmp_levels.append(lvls)

            # the batch information is always located in the first element of the list for each tissue
            # that is, the batch column vector is always the 0th column of the tensor
            if len(batch_encoded_df) > 0:
                
                tmp_k.append(batch_encoded_df)
                tmp_levels.append(batch_levels)
                tmp_combined_cat_maps.append(batch_cat_map)

            Ks.append(tmp_k)
            ks.append(tmp_levels)
            combined_cat_maps.append(tmp_combined_cat_maps)

        self.Ks, self.ks, self.cat_dict = Ks, ks, combined_cat_maps

        return None

    
    def filter_NaNs_from_COO(self, Ms):
        """
        Filters NaN from data arrays of COO (Coordinate) sparse matrices
    
        This function takes lists of COO matrices (Ms), filters out the
        NaN values from their data arrays, and returns new lists of filtered
        COO matrices.
    
        Args:
            **Ms (list):** List of COO matrices that need to be filtered

        Returns:
            **filtered_Ms (list):** List of filtered COO matrices 
    
        Notes:
            This function assumes that the input matrices are in COO format
        """
        filtered_Ms = []
        for matrix in Ms:
            data = matrix.data
            non_nan_mask = ~np.isnan(data)
            filtered_data = matrix.data[non_nan_mask]
            filtered_row = matrix.row[non_nan_mask]
            filtered_col = matrix.col[non_nan_mask]

            filtered_matrix = coo_matrix(
                (filtered_data, (filtered_row, filtered_col))
            )
            filtered_Ms.append(filtered_matrix)
        return filtered_Ms


    def pad_adjacency_matrices(self, adj_mat, final_shape):
        """
        Pads input adjacency matrix with 0s to achieve specified shape.

        **Parameters**:
        
        - **adj_mat (torch.Tensor):** 2D tensor adjacency matrix to be padded.
        
        - **final_shape (int):** The final shape that the adjacency matrix should have after padding.

        **Returns**:
       
        - **adj_mat_new (torch.Tensor):** The padded adjacency matrix with shape as specified by `final_shape`.

        **Notes**:
        
        - Padding is added to the right and bottom of the matrix.
        """
        padding_right = final_shape - adj_mat.shape[1]
        padding_bottom = final_shape - adj_mat.shape[0]

        adj_mat_new = F.pad(adj_mat, (0, padding_right, 0, padding_bottom))

        return adj_mat_new

    def from_numpy_2_tensors(self):
        """
        Converts multiple attributes from NumPy arrays to PyTorch tensors. It also clamps raw Rs, filters NaNs from As and Ss, and pads the adjacency matrices As and Ss.

        Attributes Transformed:

        - **Rs**: Expression matrices are clamped (max 100000) and converted.
        
        - **Xs**: Same as Rs for zeroed expression matrices.
        
        - **As**: Nans in Adjacency matrices are filtered.
        
        - **Ks**: Converted to tensor after concatenating dataframes.
        
        - **ks**: Converted to tensor.
        
        - **As_ej_index**: Indices of non-zeros in padded adjacencies.

        Note:

        - Assumes that self.Rs, As, and Ss are NumPy arrays or similar.
        
        - Assumes that self.Ks contains Pandas DataFrames.
        
        - Rs and Xs are logged.

        Returns:
            None. This function modifies the attributes in place.
        """
        # Convert raw Rs to tensors
        self.Rs = [torch.from_numpy(R) for R in self.Rs]
        self.Xs = [self.clamp_tensor_values(R) for R in self.Rs]

        # Convert As to tensors
        if self.configs['adj_exist']:
            self.As = self.filter_NaNs_from_COO(self.As)
            self.As_orig = [self.pad_adjacency_matrices(
                self._coo_to_dense_tensor(A), self.M) for A in self.As]
            self.As_ej_index = [torch.nonzero(A, as_tuple=False).t()
                        for A in self.As_orig]
        

 
    def _coo_to_dense_tensor(self, sparse_matrix):
        """
        Converts a sparse matrix in COO format to a dense PyTorch tensor

        Parameters:
            sparse_matrix: A sparse matrix in COO format, containing the
            attributes 'row', 'col', and 'data'.

        Returns:
            A dense PyTorch tensor representing the same matrix.

        Notes:
            Assumes 'row', 'col', and 'data' are present in sparse_matrix
        """
        indices = torch.tensor(
            np.array([sparse_matrix.row, sparse_matrix.col]), dtype=torch.long
        )
        values = torch.tensor(sparse_matrix.data, dtype=torch.int)
        size = torch.Size(sparse_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, size).to_dense()

    def clamp_tensor_values(self, tensor_list):
        """
        Clamps values of tensors in a list based on pre-defined thresholds:
        
        - Expression is clamped at >10^6.
        - Any given value less than 10^(-10) is zeroed.
        - Tensors are also converted to float.

        Parameters:
            **tensor_list (List[torch.Tensor]):** PyTorch tensors to be clamped.

        Returns:
            **torch.Tensor:** A new tensor obtained by stacking the clamped tensors along a new dimension (`dim=0`).
        """
        new_tensors = []
        for i in range(len(tensor_list)):
            tmp = tensor_list[i]
            tmp = torch.where(tmp > 1e6, torch.full_like(tmp, 1e6), tmp)
            tmp = torch.where(tmp < -1e6, torch.full_like(tmp, -1e6), tmp)
            tmp = torch.where(tmp.abs() <= 1e-10, torch.zeros_like(tmp), tmp)
            tmp = tmp.float()
            new_tensors.append(tmp)
        new_tensors = torch.stack(tensors=new_tensors, dim=0)
        return new_tensors


    def MCAR(self, arguments):
        """
        Simulates Missing Completely At Random (MCAR) by zeroing a fraction of elements
        in each tensor of the lists `self.Xs` and `self.As_orig`. The fraction of dropout 
        assigned to each gene is a random variable between 0 and the given lambda value.

        Parameters:
            arguments (dict): A dictionary containing the following keys:
                - **'lambda_counts':** The maximum dropout rate for any gene in the expression matrix.
                - **'lambda_edges':** The maximum dropout rate for the adjacency matrix.

        Returns:
            tuple: A tuple containing the following elements:
                - **List[torch.Tensor]:** A list of new tensors with zeroed elements in the expression matrix.
                - **List[torch.Tensor]:** A list of tensors indicating the indices of zeroed locations within the count tensor.
                - **List[torch.Tensor]:** A list of new tensors with zeroed elements in the edge matrix.
                - **List[torch.Tensor]:** A list of tensors indicating the indices of zeroed locations within the edge tensor.
                - **List[torch.Tensor]:** A list of tensors indicating the indices of non-zeros in masked, padded adjacencies.
        """
        As_primes, As_indices, As_ej_index_masked = [], [], []

        def _indices_to_tensor(zero_indices, shape):
            """
            Helper function to convert indices to a tensor efficiently and avoid a warning
            """
            zero_indices_2d = np.unravel_index(zero_indices, shape)
            all_indices_np = np.array(zero_indices_2d)
            return torch.from_numpy(all_indices_np)

        # zero out As_orig, and then re-create As_ej_index (rather than zero out both consistently)
        if self.configs['adj_exist']:
            for edge in self.As_orig:
                As = edge.clone()
                num_zero = int(arguments['lambda_edges'] * self.M * self.M)
                indices = np.arange(self.M * self.M)
                zero_indices = np.random.choice(indices, size=num_zero, replace=False)
                As_indices.append(_indices_to_tensor(zero_indices, (self.M, self.M)))

                As_flat = As.flatten()
                As_flat[zero_indices] = 0
                As_masked = As_flat.reshape(self.M, self.M)
                As_primes.append(As_masked)

                # Regenerate As_ej_index to masked output
                As_ej_index = torch.nonzero(As_masked, as_tuple=False).t()
                As_ej_index_masked.append(As_ej_index)

        all_indices_list, X_primes = [], []

        if self.parent.Pis is not None:
            Pis = self.parent.Pis  
            # get the prob. of a zero for each gene from the parent Preffect object
            # if this condition is true, it means that we are not in the training dataset
        else:
            Pis = torch.rand(self.N) * arguments['lambda_counts']
            self.parent.Pis = Pis

        for X in self.Xs:
            X = X.clone()  # clone X before modifying
            indices_for_all_genes = []

            for j in range(self.N):
                num_indices_to_zero = int(Pis[j] * self.M)
                indices_to_zero = random.sample(range(self.M), num_indices_to_zero)
                for idx in indices_to_zero:
                    indices_for_all_genes.append((idx, j))
                X[indices_to_zero, j] = 0
            X_primes.append(X)
            all_indices_list.extend(indices_for_all_genes)
        all_indices = torch.tensor(all_indices_list, dtype=torch.long)
        return X_primes, all_indices, As_primes, As_indices, As_ej_index_masked


    def MAR(self):
        print('to be implemented')

    def MNAR(self):
        print('to be implemented')

    def prep_batch_iterator(self, trivial_batch=False):
        """
        Splits data into multiple batches of data tensors for model training (divided by sample).

        **Parameters**:
        
        - **configs (dict):** A dictionary with the key 'mini_batch_size', specifying the number of samples/elements in each sample mini-batch.

        **Returns**:
        
        - **final_X_batches, final_R_batches, final_K_batches**: List of original expression, zeroed expression, sample information (from Obs) randomly separated into multiple "mini-batches" by sample.
        
        - **final_idx_batches**: Indices of which samples are in which minibatch.

        **Notes**:
        
        - The original tensors in `self.Xs`, `self.Rs`, and `self.Ks` are not modified.
        """

        # if mini_batch_size > M, then mini_batch_size should be M

        if self.M < self.configs['mini_batch_size']:
            self.configs['mini_batch_size'] = int(self.M)
            
        indices = torch.randperm(self.M)
        
        if trivial_batch:
            self.configs['mini_batch_size'] = int(self.M)
            indices = torch.arange(self.configs['mini_batch_size'])
        
        split_indices = indices.split(self.configs['mini_batch_size'])
        X_batches, idx_batches, R_batches, K_batches, k_batches, ej_batches, adj_batches, adj_orig_batches = [], [], [], [], [], [], [], []
        for i in range(self.calT):
            X = self.Xs[i]
            R = self.Rs[i]  
            K = self.Ks[i]
            k = self.ks[i]

            if self.configs['type'] != 'simple':
                ej = self.As_ej_index_masked[i]
                adj = self.As_mask[i]
                adj_orig = self.As_orig[i]

            X_tmp, idx_tmp, R_tmp, K_tmp, k_tmp, ej_tmp, adj_tmp, adj_orig_tmp= [], [], [], [], [], [], [], []
            for batch_indices in split_indices:
                if len(batch_indices) < self.configs['mini_batch_size']:
                    # if there is remaining samples after division by mini_batch_size
                    if self.configs['task']=='inference' or self.learning_type == "impute_experiment": 
                        tout = torch.arange(self.M)
                        rem = tout[~torch.isin(tout,batch_indices)]
                        permutation = torch.randperm(rem.size(0))
                        # later in inference, we remove this number of samples from the last minibatch
                        self.configs['number_padded'] = self.configs['mini_batch_size'] - (self.M % self.configs['mini_batch_size'])
                        batch_indices = torch.cat((batch_indices, permutation[:self.configs['number_padded']]), dim=0)
                    else:
                        continue # do nothing with the remainder during training

                X_batch = X[batch_indices, :]
                X_tmp.append(X_batch)
                idx_tmp.append(batch_indices)
                R_batch = R[batch_indices, :]
                R_tmp.append(R_batch)

                K_batch = [X[batch_indices] for X in K]
                K_tmp.append(K_batch)
                k_tmp.append(k)

                # prep induced adjacency lists
                if self.configs['type'] != 'simple':
                    mask = torch.all(torch.isin(ej, batch_indices), dim=0)
                    ej_tmp1 = ej[:, mask]
                    index_dict = {x.item(): i for i, x in enumerate(batch_indices)}
                    ej_tmp2 = torch.zeros_like(ej_tmp1)
                    for i in range(ej_tmp2.size(1)):
                        ej_tmp2[0, i] = index_dict[ej_tmp1[0, i].item()]
                        ej_tmp2[1, i] = index_dict[ej_tmp1[1, i].item()]
                    ej_tmp.append(ej_tmp2)

                    # pred induced adjacency matrices.
                    adj_tmp.append(adj[batch_indices,:][:,batch_indices])
                    adj_orig_tmp.append(adj_orig[batch_indices,:][:,batch_indices])

            X_batches.append(X_tmp)
            R_batches.append(R_tmp)
            K_batches.append(K_tmp)
            k_batches.append(k_tmp)
            idx_batches.append(idx_tmp)
            if self.configs['type'] != 'simple':
                ej_batches.append(ej_tmp)
                adj_batches.append(adj_tmp)
                adj_orig_batches.append(adj_orig_tmp)

        final_X_batches = [list(sublist) for sublist in zip(*X_batches)]
        final_R_batches = [list(sublist) for sublist in zip(*R_batches)]
        final_K_batches = [list(sublist) for sublist in zip(*K_batches)]
        final_k_batches = [list(sublist) for sublist in zip(*k_batches)]
        final_idx_batches = [list(sublist) for sublist in zip(*idx_batches)]

        if self.configs['type'] == 'simple':
            # passing an empty variable in the expected shape
            final_ej_batches = [[] for _ in range(len(idx_batches[0]))] 
            final_adj_batches = [[] for _ in range(len(idx_batches[0]))]
            final_adj_orig_batches = [[] for _ in range(len(idx_batches[0]))]
        else: 
            final_ej_batches = [list(sublist) for sublist in zip(*ej_batches)]
            final_adj_batches = [list(sublist) for sublist in zip(*adj_batches)]
            final_adj_orig_batches = [list(sublist) for sublist in zip(*adj_orig_batches)]
        
        return ( {
            'X_batches' : final_X_batches,
            'idx_batches' : final_idx_batches,
            'R_batches' : final_R_batches,
            'K_batches' : final_K_batches,
            'k_batches' : final_k_batches,
            'ej_batches' : final_ej_batches,
            'adj_batches' : final_adj_batches,
            'adj_orig_batches' : final_adj_orig_batches
        }
        )

    def to(self, device):
        self.Xs = [X.to(device) for X in self.Xs]
        # what else should be pushed to the device?
        self.Ks = [[x.to(device) for x in K] for K in self.Ks]
        # self.ks = [k.to(device) for k in self.ks]


    def imputation(self, X_hat):
        self.X_impute = X_hat.mu
        return self.X_impute

    def ground_truth(self):
        A = [None] * self.configs['calT']
        truths = []

        for i in range(self.configs['calT']):
            A[i] = self.anndatas_orig[i].copy()
            obs, var = A[i].obs, A[i].var
            truth = np.zeros(A[i].X.shape)

            if 'batch' in obs and 'mu_batch1' in var:
                batches = obs['batch']
                mu_batch1, mu_batch2 = var['mu_batch1'], var['mu_batch2']

                for j in range(A[i].n_obs):
                    if batches[j] == 0:
                        truth[j, :] = mu_batch1.to_numpy()
                    elif batches[j] == 1:
                        truth[j, :] = mu_batch2.to_numpy()

            else:
                gene_means = np.nanmean(A[i].X, axis=0)
                for j in range(A[i].n_obs):
                    truth[j, :] = gene_means

            A[i].layers['truth'] = truth.copy()
            truths.append(truth.copy())  # Append truth for this iteration

        return truths
                

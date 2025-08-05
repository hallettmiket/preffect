import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import copy
import warnings
import scanpy as sc
import anndata as ad

from _utils import (
    To,
    set_seeds,
)
from _error import ( PreffectError )

class Cluster:
    r"""
    A class for performing clustering on the latent space representation of data from a PREFFECT inference object.

    :param infer_obj: The inference PREFFECT instance. Found in the PREFFECT object (preffect_object.inference_dict[inference_key]).
    :type infer_obj: Inference
    :param configs_cluster: Configuration settings for the clustering run, which includes various operational parameters.
    :type configs_cluster: dict
    :param cluster_file_name: Optional name for the cluster file. If not provided, it is taken from the configs_cluster dictionary.
    :type cluster_file_name: str, optional
    """

    def __init__(self, infer_obj, configs_cluster, cluster_file_name=None):
        """
        Initialization of the clustering class.

        :param infer_obj: The inference PREFFECT instance. Found in the PREFFECT object (preffect_object.inference_dict[inference_key]).
        :type infer_obj: Inference
        :param configs_cluster: Configuration settings for the clustering run, which includes various operational parameters.
        :type configs_cluster: dict
        :param cluster_file_name: Optional name for the cluster file. If not provided, it is taken from the configs_cluster dictionary.
        :type cluster_file_name: str, optional
        """
        self.parent = infer_obj
        self.configs_cluster = configs_cluster
        if cluster_file_name is None:
            self.cluster_file_name = self.configs_cluster['cluster_file_name']
        else:
            self.cluster_file_name = cluster_file_name
            self.configs_cluster['cluster_file_name'] = self.cluster_file_name
        self.configs_cluster['task']='cluster'
        set_seeds(self.configs_cluster['seed'])
        self.adata = None

    def register_cluster(self):
        """
        Register the current cluster instance with the parent Inference object.

        This method checks if a cluster with the same name as ``self.cluster_file_name`` 
        is already registered in ``self.parent.clusters``. If not found, it deep-copies
        the current cluster and stores it under ``self.cluster_file_name``.

        :raises PreffectError: If the cluster name already exists and overwrite permission is set to False.
        """
        if self.cluster_file_name not in self.parent.clusters.keys() or self.configs_cluster['cluster_overwrite']:
            self.parent.clusters[self.cluster_file_name] = copy.deepcopy(self)
        else:
            raise PreffectError(f"cluster named {self.cluster_file_name} already exists and overwrite permission is False.")

    def cluster_latent_space(self, color_by='leiden', umap_nneighbors = 10, cluster_aim=5):
        """
        Extract the latent representation of the data from the parent Inference object,
        apply Leiden clustering (targeting up to 5 clusters), and visualize the results 
        using UMAP.

        This method:

        1. Retrieves an AnnData object containing the latent space representation.

        2. Constructs a neighborhood graph and computes a UMAP embedding.

        3. Iteratively reduces the Leiden resolution until five or fewer clusters are obtained (or a minimum resolution is reached).
        
        4. Plots UMAP projections colored either by the specified column (`color_by`) or, if present, by additional attributes such as `batch` or `subtype`.

        :param color_by: Column name in `adata.obs` by which to color the UMAP plot. 
                        Defaults to 'leiden'.
        :type color_by: str, optional
        :param umap_nneighbors: Number of neighbors to use for UMAP embedding. 
                                Defaults to 10.
        :type umap_nneighbors: int, optional
        :param cluster_aim: Target number of clusters to aim for during Leiden clustering. 
                            Defaults to 5.
        :type cluster_aim: int, optional
        """
        adata = self.parent.return_latent_space_as_anndata()

        adata = adata[0][:(len(adata[0].obs))]

        sc.pp.neighbors(adata, n_neighbors=umap_nneighbors, use_rep='X')  # Adjust n_neighbors based on your dataset
        sc.tl.umap(adata)

        # iterative search of the resolution parameter that yields 5 Leiden clusters
        resolution = 1

        while True:
            sc.tl.leiden(adata, resolution=resolution)
            num_clusters = adata.obs['leiden'].nunique()

            if num_clusters <= cluster_aim:
                break
            elif resolution <= 0.01:
                # set resolution to 0.1 if iterative search reduces resolution parameter to <0.01
                resolution = 0.1
                break
            elif resolution <= 0.2:
                resolution -= 0.01
            else:
                resolution -= 0.1  # decrease resolution

        # plotting UMAPs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # draw with colour of a particular correction variable (currently just one or the other)
            if 'batch' in adata.obs:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
                sc.pl.umap(adata, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title='Clustering of Latent: ' + color_by)

                adata.obs['batch'] = adata.obs['batch'].astype('category')
                adata.obs['batch'] = adata.obs['batch'].cat.add_categories('unknown')
                adata.obs['batch'] = adata.obs['batch'].fillna('unknown')

                sc.pl.umap(adata, color='batch', cmap=None, size=250, ax=axes[1], show=False, title='Clustering of Latent: Batch')
                plt.tight_layout()
                plt.show()

            if 'subtype' in adata.obs:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
                sc.pl.umap(adata, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title='Clustering of Latent: ' + color_by)
                # In case of missing data
                adata.obs['subtype'] = adata.obs['subtype'].cat.add_categories('unknown')
                adata.obs['subtype'] = adata.obs['subtype'].fillna('unknown')

                sc.pl.umap(adata, color='subtype', cmap=None, size=250, ax=axes[1], show=False, title='Clustering of Latent: Subtype')
                plt.tight_layout()
                plt.show()

            if 'subtype' not in adata.obs.columns and 'batch' not in adata.obs.columns: 
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
                sc.pl.umap(adata, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title='Clustering of Latent: ' + color_by)
                plt.tight_layout()
                plt.show()
            
        plt.tight_layout()
        plt.show()

        self.adata = adata.copy()


    def cluster_counts(self, color_by='leiden', cluster_omega = False, umap_nneighbors=10, cluster_aim=5):
        """
        Extract the estimated counts (the mu of the gene-sample NB) from the Inference object,
        apply Leiden clustering (targeting up to 5 clusters), and visualize the results
        using UMAP.

        This method:
        
        1. Retrieves an AnnData object containing the estimated counts for each gene-sample pair.
        
        2. Constructs a neighborhood graph and computes a UMAP embedding.

        3. Iteratively reduces the Leiden resolution until five or fewer clusters are obtained (or a minimum resolution is reached).        
        
        4. Plots UMAP projections colored either by the specified column (`color_by`) or, by additional attributes such as `batch` or `subtype` (if available).

        :param color_by: Column name in `adata.obs` by which to color the UMAP plot. Defaults to 'leiden'.
        :type color_by: str, optional
        :param cluster_omega: Whether to cluster the omega parameter. Defaults to False.
        :type cluster_omega: bool, optional
        :param umap_nneighbors: Number of neighbors to use for UMAP embedding. Defaults to 10.
        :type umap_nneighbors: int, optional
        :param cluster_aim: Target number of clusters to aim for during Leiden clustering. Defaults to 5.
        :type cluster_aim: int, optional
        """
        adata = self.parent.return_counts_as_anndata()

        # if you want to cluster on omega (gene-sample count divided by the sample's library size)
        if (cluster_omega):
            adata[0].X = adata[0].layers["px_omega"] 

        # set \hat{mu} 
        adata_hat = adata[0].copy()

        sc.pp.neighbors(adata_hat, n_neighbors=umap_nneighbors, use_rep='X')  # Adjust n_neighbors based on your dataset
        sc.tl.umap(adata_hat)

        # iterative search of the resolution parameter that yields 5 Leiden clusters
        resolution = 1
 
        while True:
            # Leiden currently set to set 5 clusters, if possible
            sc.tl.leiden(adata_hat, resolution=resolution)
            num_clusters = adata_hat.obs['leiden'].nunique()

            if num_clusters <= cluster_aim:
                break
            elif resolution <= 0.01:
                # set resolution to 0.1 if iterative search reaches a resolution of <=0.01
                resolution = 0.1
                break
            elif resolution <= 0.2:
                resolution -= 0.01
            else:
                resolution -= 0.1  # decrease resolution

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # samples are colour coded by batch in UMAP if this information is available
            if 'batch' in adata_hat.obs:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                if (cluster_omega):
                    sc.pl.umap(adata_hat, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title=r'Clustering of $\hat{\omega}$: ' + color_by)
                else:
                    sc.pl.umap(adata_hat, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title=r'Clustering of Counts: ' + color_by)

                # draw with colour of a particular correction variable
                # We create an "unknown" category in case of NaNs
                adata_hat.obs['batch'] = adata_hat.obs['batch'].astype('category')
                adata_hat.obs['batch'] = adata_hat.obs['batch'].cat.add_categories('unknown')
                adata_hat.obs['batch'] = adata_hat.obs['batch'].fillna('unknown')

                if (cluster_omega):
                    sc.pl.umap(adata_hat, color='batch', cmap=None, size=250, ax=axes[1], show=False, title=r'Clustering of $\hat{\omega}$: Batch')
                else:
                    sc.pl.umap(adata_hat, color='batch', cmap=None, size=250, ax=axes[1], show=False, title=r'Clustering of Counts: Batch')
                plt.tight_layout()
                plt.show()
                        
            # samples are colour coded by subtype in UMAP if this information is available 
            if 'subtype' in adata_hat.obs:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                if (cluster_omega):
                    sc.pl.umap(adata_hat, color=color_by, size=250, cmap=None, ax=axes[0], show=False, title=r'Clustering of $\hat{\omega}$: ' + color_by)
                else:
                    sc.pl.umap(adata_hat, color=color_by, size=250, cmap=None, ax=axes[0], show=False, title=r'Clustering of Counts: ' + color_by)
                # In case of missing data
                adata_hat.obs['subtype'] = adata_hat.obs['subtype'].cat.add_categories('unknown')
                adata_hat.obs['subtype'] = adata_hat.obs['subtype'].fillna('unknown')

                if (cluster_omega):
                    sc.pl.umap(adata_hat, color='subtype', size=250, cmap=None, ax=axes[1], show=False, title=r'Clustering of $\hat{\omega}$: Subtype')
                else:
                    sc.pl.umap(adata_hat, color='subtype', size=250, cmap=None, ax=axes[1], show=False, title=r'Clustering of Counts: Subtype')
                plt.tight_layout()
                plt.show()

            # Sample colour will be uniform in this UMAP
            if 'subtype' not in adata_hat.obs.columns and 'batch' not in adata_hat.obs.columns:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                if (cluster_omega):
                    sc.pl.umap(adata_hat, color=color_by, size=250, cmap=None, ax=axes[0], show=False, title=r'Clustering of $\hat{\omega}$: ' + color_by)
                else:
                    sc.pl.umap(adata_hat, color=color_by, size=250, cmap=None, ax=axes[0], show=False, title=r'Clustering of Counts: ' + color_by)
                    
                plt.tight_layout()
                plt.show()

        self.adata = adata.copy()

    # Clustering the original counts 
    def cluster_true_counts(self, color_by='leiden', umap_nneighbors=10, cluster_aim = 5):
        """
        Extract the estimated counts (the mu of the gene-sample NB) from the Inference object,
        apply Leiden clustering (targeting up to 5 clusters), and visualize the results
        using UMAP.

        This method:
        
        1. Retrieves an AnnData object containing the estimated counts for each gene-sample pair.
        
        2. Constructs a neighborhood graph and computes a UMAP embedding.
        
        3. Iteratively reduces the Leiden resolution until five or fewer clusters are obtained (or a minimum resolution is reached).
        
        4. Plots UMAP projections colored either by the specified column (`color_by`) or, by additional attributes such as `batch` or `subtype` (if available).

        :param color_by: Column name in `adata.obs` by which to color the UMAP plot. Defaults to 'leiden'.
        :type color_by: str, optional
        :param cluster_omega: Whether to cluster the omega parameter. Defaults to False.
        :type cluster_omega: bool, optional
        :param umap_nneighbors: Number of neighbors to use for UMAP embedding. Defaults to 10.
        :type umap_nneighbors: int, optional
        :param cluster_aim: Target number of clusters to aim for during Leiden clustering. Defaults to 5.
        :type cluster_aim: int, optional
        """
        adata = self.parent.return_counts_as_anndata()

        # true counts
        #adata_true = ad.AnnData(X=adata[0].layers["original_counts"])
        adata_true = adata[0].copy()
        adata_true.X = adata_true.layers["original_counts"]

        sc.pp.neighbors(adata_true, n_neighbors=umap_nneighbors, use_rep='X') 
        sc.tl.umap(adata_true)
        
        # iterative search of the resolution parameter that yields 5 Leiden clusters
        resolution = 1
        while True:
            sc.tl.leiden(adata_true, resolution=resolution)
            num_clusters = adata_true.obs['leiden'].nunique()
        
            if num_clusters <= cluster_aim:
                break
            elif resolution <= 0.01:
                # set resolution to 0.1 if iterative search doesn't find a value leading to <=5 clusters
                resolution = 0.1
                break
            elif resolution <= 0.2:
                resolution -= 0.01
            else:
                resolution -= 0.1  # decrease resolution


        # draw with colour of a particular correction variable (currently just one or the other)
        if 'batch' in adata[0].obs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sc.pl.umap(adata_true, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title=r'Clustering of True Counts: ' + color_by)
            
            # In case of missing data
            adata_true.obs['batch'] = adata[0].obs['batch'].fillna(-1).astype('category')

            sc.pl.umap(adata_true, color='batch', cmap=None, size=250, ax=axes[1], show=False, title=r'Clustering of True Counts: Batch')
            plt.tight_layout()
            plt.show()
  
        if 'subtype' in adata[0].obs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sc.pl.umap(adata_true, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title=r'Clustering of True Counts: ' + color_by)
            adata_true.obs['subtype'] = adata[0].obs['subtype'].copy()
            sc.pl.umap(adata_true, color='subtype', cmap=None, size=250, ax=axes[1], show=False, title=r'Clustering of True Counts: Subtype')
                
            plt.tight_layout()
            plt.show()

        if 'subtype' not in adata[0].obs.columns and 'batch' not in adata[0].obs.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sc.pl.umap(adata_true, color=color_by, cmap=None, size=250, ax=axes[0], show=False, title=r'Clustering of True Counts: ' + color_by)
                
            plt.tight_layout()
            plt.show()

[← Back to Main](../readme/readme.md#clustering)

# Clustering
PREFFECT provides clustering functions to evaluate UMAP clustering of the latent space or the estimated counts of a PREFFECT model

To load a pre-trained model into your Python session.
```python
    from preffect_factory import factory
    from _config import configs

    configs['output_path'] = "/location/of/PREFFECT/Model" # Point to the PREFFECT model you wish to import
    preffect_object_reinstated = factory(task='reinstate', configs=configs, trigger_setup=True)
```

Then, you can generate UMAPs of counts (both observed and estimated), as well as the latent space with the following commands:

```python
# Cluster counts estimated by the PREFFECT model
factory(task='cluster_counts', preffect_obj=preffect_object_reinstated, inference_key='endogenous', trigger_setup=False, configs=configs)
# Cluster counts used as input when training the PREFFECT model (observed or "true" counts)
factory(task='cluster_true_counts', preffect_obj=preffect_object_reinstated, inference_key='endogenous', trigger_setup=False, configs=configs)
# Cluster the latent space of expression
factory(task='cluster_latent', preffect_obj=preffect_object_reinstated, inference_key='endogenous', trigger_setup=False, configs=configs)
```
Change `inference_key` if you wish to perform this evaluation on a different Inference task (e.g. batch-adjusted estimated counts).

Two side-by-side UMAPs will be created. The left-most will be colour-coded based on Leiden clustering. The right-most UMAP will be colour coded by either batch or subtype (if provided in the input AnnData obs structure as 'batch' or 'subtype'). 

By default, the UMAP parameters (e.g. `n_neighbors`) are set to 10. Furthermore, we tune the "resolution" parameter of Leiden clustering to attempt to find a value which yields 5 clusters. While you cannot tune these parameters through `factory()`, you can change these parameters by importing the `Cluster()` class used by `factory()`:


```python
from wrappers._cluster import( Cluster )

inference_object = preffect_object_reinstated.inference_dict[inference_key] 
cl = Cluster(infer_obj=inference_object, configs_cluster=configs )

# `umap_nneighbors` sets UMAP n_neighbors parameter
# `cluster_aim` sets the desired number of Leiden clusters by adjusting `resolution` parameter (if not found, parameter set to 0.1)
cl.cluster_latent_space(color_by = "leiden", umap_nneighbors=10, cluster_aim=5)
cl.cluster_counts(color_by = "leiden", umap_nneighbors=10, cluster_aim=5)
```

### Clustering the Omega Parameter (Count / Library Size)

To reduce the influence of differentiating library sizes, you can set the cluster_counts() function to cluster by $\omega$ (gene count divided by the library size of the sample) using the `cluster_omega` boolean parameter.

```python
from wrappers._cluster import( Cluster )

inference_object = preffect_object_reinstated.inference_dict[inference_key] 
cl = Cluster(infer_obj=inference_object, configs_cluster=configs )

# `umap_nneighbors` sets UMAP n_neighbors parameter
# `cluster_aim` sets the desired number of Leiden clusters by adjusting `resolution` parameter (if not found, parameter set to 0.1)
cl.cluster_counts(color_by = "leiden", umap_nneighbors=10, cluster_aim=5, cluster_omega=True)
```

##
[← Back to Main](../readme.md#clustering)
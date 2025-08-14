[← Back to Main](../readme/readme.md#clustering)

# Clustering
PREFFECT provides clustering functions to evaluate UMAP clustering of the latent space or the estimated counts of a PREFFECT model.

You must first import preffect_factory, and configuration settings (if not set from within the script itself).
```python
    from preffect_factory import factory
```

If training was not performed within your session and you wish to import a previously trained model, we must use the 'reinstate' task.

```python
    # Create a factory object where 'output_path' is the path to your PREFFECT model
    preffect = factory(                      
            type='simple', # match the conditions of the model
            ...
            output_path = "/path/to/PREFFECT/Model/" 
    )
    _ = preffect.reinstate(fname=None)
```


Then, you can generate UMAPs of counts (both observed and estimated), as well as the latent space with the following commands:

```python
# Cluster counts estimated by the PREFFECT model
fac.visualize_embedding(mode='counts', ir_name=INFERENCE_NAME)

# Cluster counts used as input when training the PREFFECT model (observed or "true" counts)
fac.visualize_embedding(mode='true_counts', ir_name=INFERENCE_NAME)

# Cluster the latent space of expression
fac.visualize_embedding(mode='latent', ir_name=INFERENCE_NAME)
```

Change `ir_name` if you wish to perform this evaluation on a different Inference task (e.g. batch-adjusted estimated counts).

This will generate a single UMAP, which will be colour coded by either batch or subtype (if provided in the input AnnData `obs` structure as _batch_ or _subtype_). 

You can also preform Leiden clustering (identifies groups of nodes more densely connected to each other than to the rest of the network) by setting `perform_leiden` to true. This is off by default. When active, two UMAPs will be generated. The left-most will be colour-coded based on Leiden clustering. The right-most UMAP will be colour coded by either batch or subtype (if provided in the input AnnData `obs` structure as _batch_ or _subtype_). If an optional parameter `cluster_aim` (integer) is used, Leiden will attempt to identify `cluster_aim` number of clusters (though this is not always successful).

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

To reduce the influence of differentiating library sizes, you can set the `cluster_counts()` function to cluster by $\omega$ (gene count divided by the library size of the sample) using the `cluster_omega` boolean parameter.

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
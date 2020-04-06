# mlclust
A package for preprocessing, projecting, visualizing, building binary classifers, and clustering data. 

# Getting Started
git clone https://github.com/atfranc2/mlclust.git

# Parameters

**standard_method: str, default = None**
    
    Whether to standardize the data prior to analysis
  
    Options: [None, 'standardize', 'min_max', 'center']
    
**projection_method: str, default = None**
    
    Specifies if a projection techniqe is to be applied to the data prior to analysis
  
    Options: [None, 'pca', 'umap', 'tsne', 'phate', 'isomap']
    
**select_var_method: str, default = None**

    Specifies a way to select significant and important variables from a dataset
    
    Options: [None, 'lasso', 'pvalue']
    
    
**n_comps: int, default = None**

    Specifies the number of components/variables to use from the projected/raw data after projection, but prior to variable selection

    Note 1: If no projection method is specified then this option will select the 1:n_comps variables from the raw data
    
    Note 2: If a projection method is specified and n_comps = None then the maximum allowable components is selected
    
    
**n_neighbors: int, default = None**

    If umap or tsne is the projection method then this controls the n_neighbors parameter. Otherwise this parameter is ignored
    
    Note 1: For umap n_neighbors <= n_comps
    
**rand_state: int, default = 123**

    The random seed to use in any randomized operations
 
 

# Methods

**define_vars**(self, x_df, y_df, alpha = 0.05)

    Executes all of the data preprossessing and variable selection opterations
    
    Parameters:
    x_df: Pandas Dataframe 
        Contains the predictor variables
    
    y_df: Pandas Dataframe 
        Contains the target variable (only binary categorical targets are currently supported)
    
    alpha: float
        If the pvalue method was specified in select_var_method then an alpha level must be set to distiguish Significant variables.
        Otherwise this option is ignored.
        

**define_vars**(self, x_df, y_df, alpha = 0.05)

    Executes all of the data preprossessing and variable selection opterations
    
    Parameters:
    x_df: Pandas Dataframe 
        Contains the predictor variables
    
    y_df: Pandas Dataframe 
        Contains the target variable (only binary categorical targets are currently supported)
    
    alpha: float
        If the pvalue method was specified in select_var_method then an alpha level must be set to distiguish Significant variables.
        Otherwise this option is ignored.


**explore_kmclusters**(self, cluster_try)

    cluster_try: int, default = 10
        The number of cluster centroids to explore in the kmeans algorithm


**explore_dbclusters**(self, lower_eps_quant, upper_eps_quant, eps_breaks, sample_set)

    # Lower quantile of eps values to use in DBSCAN
    lower_eps_quant = 0.05,
    
    # Upper quantile of eps values to use in DBSCAN
    upper_eps_quant = 0.35, 
    
    # How many eps values to try
    eps_breaks = 10,
    
    # What sample sizes to try to be considered a cluster
    sample_set = [5,10,15] 










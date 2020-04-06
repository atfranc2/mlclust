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


**explore_kmclusters**(self, cluster_try)

    cluster_try: int, default = 10
        The number of cluster centroids to explore in the kmeans algorithm


**explore_dbclusters**(self, lower_eps_quant, upper_eps_quant, eps_breaks, sample_set)
    
    Computes the minimum and maximum distance bewteen any two points in the dataset. Then breaks the interval
    down into equal sized break to try as the eps value in the DBSCAN algorithm. 

    lower_eps_quant: float
        Lower quantile of eps values to use in DBSCAN. This is the quantile of dsitance values between 
        points in the dataset.
   
    upper_eps_quant: float
        Upper quantile of eps values to use in DBSCAN. This is the quantile of dsitance values between 
        points in the dataset.
    
    eps_breaks: int, default = 10
         How many eps values to try
    
    sample_set: list, default = [5,10,15] 
        The values of min_samples to try in SKLearns DBSCAN algorithm. 

**fit_clusters**(self, cluster_method, num_clust, eps, min_samples)

    cluster_method: str
        The clustering algorthm to use to cluster points
        Options: ['kmeans', 'dbscan']
    
    num_clust: int
        The number of cluster centroids to initialize when 'kmeans' is specifed as the cluster_method. 
        Otherwise it is ignored.
    
    eps: float
        The eps value to use in SKLearns DBSCAN algorthm if 'dbscan' is the method specified in cluster_method. Otherwise it is               ignored.
    
    min_samples: int
        The min_samples value to use in SKLearns DBSCAN algorthm if 'dbscan' is the method specified in cluster_method. 
        Otherwise it is ignored.

**plot_2dclusters**(self, proj_method)
    
    Plot the data projected into 2-D space
    
    Parameters: 
    
    proj_method: str
        If a projection method is not specified then the data must be projected prior to viewing it is 2-D. 
        This option specifes the method to project the data. If a projection_method is specifed in __init__
        then it is used by default. 
        Options:['pca', 'umap', 'tsne', 'phate', 'isomap']


# Usage Example

Get some data: 

    toy = df[df.columns[0:20]].iloc[0:5000]

    target = df['target1'].iloc[0:5000]

Initialize the mlclust object class: 

    mlclust_object = mlc.Cluster(

        # How to standardize the data prior to perfroming any other analysis. 
        # Options: [None, 'standardize', 'min_max', 'center']
        standard_method = 'standardize', 

        # How to project the data prior to varible selection or clustering. 
        # Options: [None, 'pca', 'umap', 'tsne', 'phate', 'isomap']
        projection_method = 'pca', 

        # How to select significant variables before clustering.
        # Options: ['lasso', 'pvalue']
        select_var_method = 'pvalue', 

        # How many components to fit using the specified projection method. 
        # Options: [None, int]
        # Note 1: If no projection method is specified then this option will select the 1 - n_comps variables from the raw data
        # Note 2: If a projection method is specified and n_comps = None then the maximum allowable components
        n_comps = None, 

        # If umap or tsne is the projection method then this controls the n_neighbors parameter
        # Options = [int]
        # Note 1: For umap n_neighbors <= n_comps
        n_neighbors = 100, 

        # Random state for any randomized operations
        rand_state = 123

    )
    
Preprocess the data by standardizing the data -> projecting the data using PCA -> Selecting principal components
that are significant in predicting the target classes at alpha = 0.05: 

    mlclust_object.define_vars( 

        # Specify the predictor variables (i.e. the spectral variables)
        x_df = toy, 

        # Specify the target variable
        # Note 1: Only binary target variables are supported
        y_df = target, 

        # If the pvalue method was specified in select_var_method then an alpha level must be set to distiguish 
            # Significant variables
        # Note 1: This should be set to the metabolome wide significance level
        alpha = 0.05 
    )

Pull significant variables: 
    
    mlclust_object.signif_vars
    
    Output: array(['PC18', 'PC2', 'PC12', 'PC19', 'PC3'], dtype=object)
    
Pull the project object. In this case it is an SKlearn PCA object: 

    mlclust_object.project_obj.explained_variance_ratio_[0:10]
    
    Output: array([0.1796708 , 0.17337912, 0.08249129, 0.07046907, 0.06280863,
                   0.06127295, 0.05087925, 0.04469096, 0.03903693, 0.03445266])
                   
Get the dataframe with the significant pricnipal components sorted by decreasing significance: 

    mlclust_object.cluster_data.head()
    
    Output: 
    
|   | PC18      | PC2       | PC12      | PC19      | PC3       |
|---|-----------|-----------|-----------|-----------|-----------|
| 0 | -0.064003 | 0.486440  | -0.613948 | -0.447888 | 1.083790  |
| 1 | -0.064813 | -1.145281 | 0.369435  | 0.122299  | 1.442297  |
| 2 | -0.047029 | 1.454269  | 0.631783  | 0.162778  | -1.360187 |
| 3 | 0.039484  | -0.926692 | 0.739888  | 0.170727  | -0.080960 |
| 4 | -0.136652 | -2.704280 | 0.234936  | 0.287860  | -0.992221 |


Plot a kmeans elbow plot: 

    mlclust_object.explore_kmclusters( 

        # How many cluster centroid to try
        cluster_try = 10 
    )
    



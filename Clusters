import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.decomposition import PCA
import umap
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from . import Standardize, Projection, VariableSelection, difftest



class Cluster( VariableSelection.VariableSelect, difftest.DiffTests ):
    
    
    
    def __init__( self, standard_method = None, projection_method = None, select_var_method = None, 
                  n_comps = None, n_neighbors = 100, rand_state = 123 ):
        
        self.rand_state = rand_state
        
        self.standard_method = standard_method
        
        self.projection_method = projection_method
        
        self.select_var_method = select_var_method
        
        self.n_comps = n_comps
        
        self.n_neighbors = n_neighbors
        
        self.cluster_method_id = None
        
        self.input_data = None
        
        self.cluster_object = None
        
        self.cluster_data = None
            
        self.labels = None
        
        self.orig_data = None
        
        self.alpha = None
        
        self.reference_data = None
        
        self.signif_vars = None
        
        self.pca_fitted = False
        
        self.standard_data = None
        
        self.standard_obj = None
        
        self.project_obj = None
        
        self.project_data = None
        
        super(  ).__init__( standard_method, projection_method, n_comps = n_comps, n_neighbors = n_neighbors )
        
    
    
    def define_vars( self, x_df, y_df, alpha = None ):
        
        '''
        
        This will apply variable the selection methods specified in the VarSelect class
        
        '''
        
        assert self.select_var_method in ['lasso', 'pvalue', 'n', None], 'Not a viable variable selection method'
        
        if self.select_var_method == 'pvalue':
            
            assert alpha != None, 'Alpha level must be specifed to select variables based on pvalue'
            
            cluster_vars = self.pvalue_variable_selection( x_df, y_df, alpha, return_vars = True )
            
            cluster_data = self.reference_data[ cluster_vars ]
            
        elif self.select_var_method == 'lasso':
            
            cluster_vars = self.lasso_variable_selection( x_df, y_df, rand_seed = self.rand_state, return_vars = True )
            
            cluster_data = self.reference_data[ cluster_vars ]
            
        elif self.select_var_method == 'n':
            
            assert n_comps != None, 'Set n_comps to an integer specifying the number of variables or components you want to \
                                     use in the analysis'
                
            cluster_data = self.return_data( x_df )
            
            cluster_data = cluster_data[ cluster_data.columns[0:n_comps] ]
            
        else: 
            
            cluster_data = self.data_preprocessing( x_df )
            
        self.cluster_data = cluster_data
            
        #return cluster_data
        
    
    
    def explore_kmclusters( self, cluster_try = 10 ):
        
        '''
        
        Will create an elbow plot showing the reduction in the sum of squared distances between cluster 
        centroids and the data points assigned the them as the number of clusters is increased.
        
        '''
        
        sil_score_max = -1

        cluster_tries = [ i for i in range( 2, cluster_try + 1 ) ]

        ssd = [ ] 

        for n_clust in cluster_tries: 

            km = KMeans( n_clusters = n_clust, random_state = self.rand_state ).fit( self.cluster_data )

            ssd.append( km.inertia_ )
            
            labels = km.predict( self.cluster_data )

            sil_score = silhouette_score( self.cluster_data, labels)

            # Store the best silhouette score so that we can automate running another kmeans later with the 'most optimal' number of clusters

            if sil_score > sil_score_max:

                sil_score_max = sil_score

                best_n_clusters = n_clust

                
        plt.axvline( best_n_clusters, linestyle='--', color='black' )
        
        plt.legend(['Silhouette Score Recommended Clusters'])
        
        plt.plot( cluster_tries, ssd )

        plt.scatter(  cluster_tries, ssd )

        plt.xlabel( 'Clusters' )

        plt.ylabel( 'Sum of Squared Distances' )
        
        
        
    def explore_dbclusters( self, lower_eps_quant = 0.05, upper_eps_quant = 0.35, eps_breaks = 10, sample_set = [5,10,15] ):
        
        import sklearn.metrics as met
        
        dists = met.pairwise_distances( self.cluster_data )

        flat_dists = dists.flatten()

        flat_dists = flat_dists[ flat_dists != 0 ]
        
        min_dist = np.quantile( flat_dists, lower_eps_quant )
        
        max_dist = np.quantile( flat_dists, upper_eps_quant )
        
        epss = np.linspace( min_dist, max_dist, eps_breaks )

        noise = []

        clusts = []

        for samp in sample_set:

            samp_noise = []

            samp_clusts = []

            for eps in epss:

                db = DBSCAN( eps = eps, min_samples = samp ).fit( self.cluster_data )

                labels = db.labels_

                noise_prop = len(labels[ labels == -1 ]) / len( labels )

                samp_noise.append( noise_prop )

                not_noise = labels[ labels != -1 ]

                samp_clusts.append( len( np.unique( not_noise ) ) )

            noise.append( samp_noise )

            clusts.append( samp_clusts )

        plt.figure( figsize=(10,5))

        plt.subplot( 1,2,1 )

        for n in noise: 

            plt.scatter( epss, n )

            plt.plot( epss, n )

            plt.xlabel( 'eps' )

            plt.ylabel( 'Proportion Noise' )

        plt.legend( sample_set, title = 'Min Samples' )

        plt.subplot( 1,2, 2 )

        for c in clusts: 

            plt.scatter( epss, c )

            plt.plot( epss, c )

            plt.xlabel( 'eps' )

            plt.ylabel( 'Clusters' )

        plt.legend( sample_set, title = 'Min Samples' )

        plt.show()
        
        print('The eps values tried were', epss )
    
    
           
    def fit_clusters( self, cluster_method = 'kmeans', num_clust = 2, eps = 2, min_samples = 3 ):
        
        '''
        
        Will fit clusters to the data using the method specified in cluster_method. Returns nothing but
        saves the cluster labels (i.e. labels) and cluster object (i.e. cluster_object) to the class.
        
        '''
        
        assert type(self.cluster_data) != type(None), 'Clustering data has not been defined. Make sure to run the define_vars() method \
                                            prior to running this method' 
        
        assert cluster_method in ['kmeans', 'dbscan'], 'Not a valid clustering method'
        
        
        if cluster_method == 'kmeans':
            
            assert num_clust > 1, 'For kmeans the number of clusters (num_clust) must be a number greater than 1'
            
            self.cluster_method_id = cluster_method
            
            self.cluster_object = KMeans( n_clusters = num_clust, random_state = self.rand_state ).fit( self.cluster_data )
            
            self.labels = self.cluster_object.labels_
            
        elif cluster_method == 'dbscan':
            
            self.cluster_method_id = cluster_method
            
            self.cluster_object = DBSCAN( eps = eps, min_samples = min_samples ).fit( self.cluster_data )
            
            self.labels = self.cluster_object.labels_
            
            
                
    def plot_2dclusters( self, proj_method = 'pca' ):
        
        '''
        
        Will plot the data in two dimensions by performing the projection_method, taking the first two components, 
        and then coloring the data by their respective clusters. 
        
        If the pre_project_method specified in the __init__ function was not set then pca is performed by default.
        However, if pre_project_method is not set all of the supported pre_project_methods can be used by setting 
        the projection_method local to this class equal to your desired projection method. All of the supported 
        pre_project_methods can be used when setting projection_method.
        
        If pre_project_method is set the the first two components of the projection are used for the two dimensional
        representation. However, all of the supported pre_project_methods can be used. 
        
        '''
        
        if self.projection_method == None:
            
            comps = self.fit_transform_pca( self.reference_data, n_comps = 2, return_data = True )
            
            comps = pd.DataFrame( comps )
            
        else: 
            
            proj_method = self.projection_method
            
            cols = self.reference_data.columns
            
            comps = self.reference_data[ cols[ 0:2 ] ]

        plt.scatter( comps[ comps.columns[0] ], comps[ comps.columns[1] ], c = self.labels, cmap = 'Set1' )

        plt.xlabel( proj_method + '1' )

        plt.ylabel( proj_method + '2' )
        
    
    
    # Perform significance test on clusters
    def cluster_analysis(self, target_col, group_col, test = 'Fisher', group_alpha = 0.01, pairwise_alpha = 1.88e-6, 
                         pairwise_method = 'median', plot = True, return_data = False):
        
        assert test in ['Fisher', 'ANOVA'], 'target_type must be binary, ordinal, or nominal' 
        
        df = self.input_data.copy(deep=True)
        
        cols = self.input_data.columns
        
        #df['Target'] = target_col

        df['Group'] = group_col 
        
        
        
        if test == 'Fisher':
            
            contingency_table = self.chi(target_col, group_col, group_alpha)
            
            sig_combos = self.fisher(df = df[cols], 
                                     contingency_table=contingency_table, 
                                     group_col = group_col, 
                                     group_alpha = group_alpha, 
                                     pairwise_alpha = pairwise_alpha,
                                     pairwise_method = pairwise_method)
            
            
            #self.plot_fisher_groups(df, sig_combos, index_name = 'Group', var_cols = cols)
            
        elif test == 'ANOVA':
            
            sig_combos = self.ANOVA(df = df[cols], target_col = target_col, group_col = group_col, 
                                    pairwise_method = pairwise_method, group_alpha = group_alpha, 
                                    pairwise_alpha = pairwise_alpha )
        
        
        
        if plot == True:
            self.plot_fisher_groups(df, sig_combos, index_name = 'Group', var_cols = cols, alpha = pairwise_alpha)
                    
        if return_data == True: 
            return sig_combos
        
        
        
        

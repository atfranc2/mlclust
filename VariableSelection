'''

Here we will consolidate all of our important variable extraction tools into a comprehensive class: 

*********************************************************
*** Currently only support binary logistic regression ***
*********************************************************

#########
Arguments
##########

spectral_feature_df = A pandas dataframe containing only spectral variables. Variable by column and observation by row

target_df = A pandas dataframe containing only the target variable associated with spectral_feature_df

methods = 

    - significance: Will use an alpha level and significance level to extract significant variables
    
    - lasso: Will perform a regularlized lasso regression 
    
    
    
    
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr
from . import Standardize, Projection


'''

Here we will consolidate all of our important variable extraction tools into a comprehensive class: 

*********************************************************
*** Currently only support binary logistic regression ***
*********************************************************

#########
Arguments
##########

spectral_feature_df = A pandas dataframe containing only spectral variables. Variable by column and observation by row

target_df = A pandas dataframe containing only the target variable associated with spectral_feature_df

methods = 

    - significance: Will use an alpha level and significance level to extract significant variables
    
    - lasso: Will perform a regularlized lasso regression 
    
    
    
    
'''

class VariableSelect( Projection.Project ): 
    
    '''
    
    The VariableSelect class allows variables to be selected using p-value significance or lasso variable importance.
    Currently this method only supports binary categorical target variables. 
    
    VariableSelect inherits methods from Standard and Project so that the data can be standardized and projected
    prior to selecting variables. If the data is projected then components of the projection are the variables that 
    are selected. 
    
    '''
    
    def __init__( self, standard_method = None, projection_method = None, n_comps = None, n_neighbors = 100 ):
        
        '''
        
        Parameters
        ----------
        
        standard_method: str, default = None
            The standardization method to employ prior to selecting variables. Valid options are 'standardize',
            'center', and 'min_max'
            
        projection_method: str, default = None
            The projection method to used before selecting variables. Valid options are 'pca' and 'umap'
            
        n_comps: int, default = None
            The number of components to consider for variable selction. If n_comps is not None then variables
            are taken from left to right from the underlying projected dataset ( i.e. components 1 through n_comps are
            selected )
        
        n_neighbors: int, default = 100
            If projection_method = 'umap' then n_neighbors will define the size of the local neighborhood to 
            consider when constructing the manifold. Note that a binding rule is that n_comps <= n_neighbors
        
        '''
        
        self.standard_method = standard_method
        
        self.projection_method = projection_method
        
        self.n_comps = n_comps
        
        self.n_neighbors = n_neighbors
        
        self.input_data = None
        
        self.alpha = None
        
        self.reference_data = None
        
        self.signif_vars = None
        
        self.pca_fitted = False
        
        self.umap_fitted = False
        
        self.standard_data = None
        
        self.standard_obj = None
        
        self.project_obj = None
        
        self.project_data = None
        
        super(  ).__init__( standard_method )
        
        
        
    def reset_params( self ):
        
        self.input_data = None
        
        self.alpha = None
        
        self.reference_data = None
        
        self.signif_data = None
        
        self.pca_fitted = False
        
        self.umap_fitted = False
        
        self.standard_data = None
        
        self.standard_obj = None
        
        self.project_obj = None
        
        self.project_data = None
        
        return self
    
    
    
    def data_preprocessing( self, x_df ):
        
        '''
        
        This funtion will perform all of the specified standardization and projection procedures and then
        return the transformed/projected dataset.
        
        Parameters
        ----------
        x_df: pandas DataFrame
            The pandas DataFrame containing the relavent variables and their associated column names
        
        '''
        
        self.input_data = x_df
        
        
        
        if (self.projection_method == None) & (self.standard_method != None):
            
            cols = x_df.columns
            
            x_df = self.fit_transform_standard( x_df, return_data = True )
            
            x_df = pd.DataFrame( x_df, columns = cols )
        
        
        
        elif self.projection_method == 'pca':
            
            x_df = self.fit_transform_pca( x_df, n_comps = self.n_comps, return_data = True )
            
            cols = ['PC'+str(i) for i in range( 1, x_df.shape[-1] + 1 )]
            
            x_df = pd.DataFrame( x_df, columns = cols )
        
        
        
        elif self.projection_method == 'umap':
            
            x_df = self.fit_transform_umap( x_df, n_comps = self.n_comps, n_neighbors = self.n_neighbors, return_data = True )
            
            cols = ['UMAP'+str(i) for i in range( 1, x_df.shape[-1] + 1 )]
            
            x_df = pd.DataFrame( x_df, columns = cols )
        
        
        
        elif self.projection_method == 'tsne':
            
            x_df = self.fit_transform_tsne( x_df, n_comps = self.n_comps, return_data = True )
            
            cols = ['TSNE'+str(i) for i in range( 1, x_df.shape[-1] + 1 )]
            
            x_df = pd.DataFrame( x_df, columns = cols )
            
        
        
        elif self.projection_method == 'phate':
            
            x_df = self.fit_transform_phate( x_df, n_comps = self.n_comps, return_data = True )
            
            cols = ['PHATE'+str(i) for i in range( 1, x_df.shape[-1] + 1 )]
            
            x_df = pd.DataFrame( x_df, columns = cols )
            
            
            
        elif self.projection_method == 'isomap':
            
            x_df = self.fit_transform_isomap( x_df, n_comps = self.n_comps, n_neighbors = self.n_neighbors, 
                                              return_data = True )
            
            cols = ['ISOMAP'+str(i) for i in range( 1, x_df.shape[-1] + 1 )]
            
            x_df = pd.DataFrame( x_df, columns = cols )
        
        
        
        self.reference_data = x_df
        
        self.plot_cols = x_df.columns
            
        return x_df
    
    
    
    def pvalue_variable_selection( self, x_df, y_df, alpha, return_vars = False ): 
        
        '''
        
        Performs pvalue based variable selection by considering each of the variables significance 
        independently in a logistic regression model with an intercept. 
        
        Parameters
        ----------
        
        x_df: Pandas DataFrame
            The predictor variables to consider in the variable selection technique. The dataset that is input
            with be standardized, projected, and sized (i.e. n_comps) according to the parameters set
            in __init__() prior to performing variables selection. 
            
        y_df: Pandas DataFrame (or Series) 
            The target variable to consider when assessing the significance of each variable in x_df. 
            *** Currently only binary categorical target variables are supported
            
        alpha: int or float
            The alpha level to use when assesing the significance of each variable. Variables with pvalues < alpha
            will be selected.
            
        return_vars: bool, default = False
            If return_vars is True then a numpy array containing the names of the significant variables 
            that were found will be returned. 
        
        
        '''
        
        x_df = self.data_preprocessing( x_df )

        var_df = sm.add_constant( x_df )

        cols = x_df.columns

        p_values = []

        for i in range( 1, len( cols ) ): 

            #Create a model object using the training data 
            model = sm.Logit( y_df, var_df[[ cols[ 0 ], cols[ i ] ]] )

            #Fit the model to the trainind data fold
            model = model.fit( disp = 0 )

            signif = model.pvalues.values[1]

            p_values.append( [ cols[i], signif ])

        p = pd.DataFrame( p_values )

        signif_vars = p[ p[1] <= alpha ]

        signif_vars = signif_vars.sort_values( by = 1 )
        
        self.signif_vars = np.array( signif_vars )[:,0]
        
        if return_vars == True:
            return self.signif_vars
    
    
    
    def lasso_variable_selection(self, x_df, y_df, return_vars = False, rand_seed = 1234 ):  
    
        #Set a random seed which will make logistic regression results consistent
        np.random.seed( rand_seed )
                 
        x_df = self.data_preprocessing( x_df )
        
        cols = x_df.columns
        
        #Add a constant to the data
        var_df = sm.add_constant( x_df )
        
        #Create a model object 
        LR = lr(max_iter = 300, penalty = 'l1', C = 0.1, tol=0.001, solver='liblinear')
        
        #Fit the model to the trainind data fold
        model = LR.fit( var_df, y_df )
        
        # Get the coefficients
        coeff_size = pd.DataFrame(abs(LR.coef_), columns = var_df.columns ).transpose()
        
        # Make the coefficients a dataframe
        variable_importance = pd.DataFrame( coeff_size ).drop( 'const' )
        
        # Reset Index
        variable_importance = variable_importance.reset_index()
        
        # Define Column Names
        variable_importance.columns = ['Variable', 'Coefficient']
        
        # Sort
        variable_importance = variable_importance.sort_values(by='Coefficient', ascending=False)
        
        # Find the top 25th percentile in terms of coefficient size
        #top_25_vars = coeffs[coeffs.iloc[:,1]>np.percentile(coeffs.iloc[:,1], 75)]
        
        # Only want the variables above 0
        nonzero_vars = variable_importance[variable_importance.iloc[:,1] > 0]
        
        # Make a list of the important variables
        self.signif_vars = nonzero_vars.iloc[:,0].values
        
        if return_vars == True:
            return self.signif_vars


        
    def plot_sig_vars( self ):
        
        cols = self.reference_data.columns
        
        col_map = pd.DataFrame( [[i for i in range(0, len( cols ))]], columns = cols )
        
        ids = col_map[ self.signif_vars ].values
        
        ids = np.sort(ids)[0]
        
        start = 0 

        intervals = []

        for i in range(0, len( ids ) - 1 ): 

            diff = ids[i+1] - ids[i]

            if diff > 1: 

                intervals.append( list( ids[start:i+1] ) )

                start = i+1

            if i == len( ids ) - 2  :

                intervals.append( list( ids[start:i+2] ) )

        y = self.input_data.median()

        plt.plot( [ i for i in range (0, len( self.input_data.columns ) ) ], y )

        for i in intervals: 
            
            plt.axvspan(min(i), max(i), color='red', alpha=0.5)
        
        plt.show()

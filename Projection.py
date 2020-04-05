from . import Standardize
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import phate
from sklearn.manifold import Isomap


class Project( Standardize.Standard ):

    
    '''
    
    The Project class will project data in to new feature space using linear and non-linear transformation methods.
    
    The currently supported projections are PCA and UMAP.
    
    '''
    
    def __init__( self, standard_method = None ):
        
        '''
        The class take as input the valid standardization methods from the Standard class. Valid standardization 
        methods include:
        
            - 'standardize' = Mean center the data and scale by the standard deviation
            
            - 'center' abs  = Mean center the data
            
            - 'min_max'     = Scale the data to between 0 and 1
        
        '''
        
        self.standard_method = standard_method
        
        self.pca_fitted = False
        
        self.umap_fitted = False
        
        self.standard_data = None
        
        self.standard_obj = None
        
        self.project_obj = None
        
        self.project_data = None
        
        super(  ).__init__( standard_method )
    
    
    
    def reset_params( self ):
        
        '''
        
        Reset the standization and projection parameters to their default values
        
        '''
        
        self.standard_fitted = False
        
        self.pca_fitted = False
        
        self.standard_data = None
        
        self.standard_obj = None
        
        self.project_obj = None
        
        self.project_data = None
        
        return self
            
            
    def fit_pca( self, x, n_comps = None ):
        
        '''
        
        Fit a pca object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the pca object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut None, n_comps <= min( n_features, n_observations ) 
                       Specifies the number of principal components to fit to the dataset
        
        '''
        
        if n_comps == None:
            
            n_comps = min( x.shape )
        
        assert n_comps <= min( x.shape ), 'n_comps must be less than or equal to the minimum element of x.shape'
        
        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = PCA( n_components = n_comps ).fit( x )
        
        self.pca_fitted = True
        
        return self.project_obj
        
        
        
    def transform_pca( self, x = None, return_data = False ):
        
        '''
        
        Project a dataset using the pca object fitted in the fit_pca() method: 
        
            - x: Array or dataframe, default = None, optional  
                 Contains the data to be transformed by the pca object. If x is None then the data used to fit 
                 the pca object in fit_pca() will be used in transfrom_pca. 
                 
                 If x is not None the the fitted parameters of the pca object will be used to transform x. 
                 Furthermore, if a standardization method is specified and x is not None then the parameters of
                 the standardization object fitted in fit_pca() will be used to standardize x before proejcting 
                 the data.
            
            - return_data: bool, default = False
                 If True the function will return the projected dataset as a numpy array
        
        '''
        
        
        assert self.pca_fitted == True
        
        
        if self.standard_method != None:
        
            if type( x ) != type( None ): 

                x = self.transform_standard( x, return_data = True )

            else: 

                x = self.standard_data
        
        
        self.project_data = self.project_obj.transform( x )
        
        if return_data == True: 
            return self.project_data
        
    
    def fit_transform_pca( self, x, n_comps = None, return_data = False ):
        
        '''
        
        Fit a pca object and project the dataset using the fitted pca object with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the pca object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut None, n_comps <= min( n_features, n_observations ) 
                       Specifies the number of principal components to fit to the dataset
                        
            - return_data: bool, default = False
                 If True the function will return the projected dataset as a numpy array
        
        '''
        
        if n_comps == None:
            
            n_comps = min( x.shape )
        
        assert n_comps <= min( x.shape ), 'n_comps must be less than or equal to the minimum element of x.shape'
        
        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = PCA( n_components = n_comps ).fit( x )        
        
        self.project_data = self.project_obj.transform( x )
        
        self.pca_fitted = True
        
        if return_data == True: 
            return self.project_data
        
        
        
    def fit_umap( self, x, n_comps = None, n_neighbors = 50 ):
                
        '''
        
        Fit a UMAP object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the pca object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                Specifies the number of UMAP components to fit to the dataset
            
            - n_neighbors: int, default = 50
                This parameter controls how UMAP balances local versus global structure in the data. 
                It does this by constraining the size of the local neighborhood UMAP will look at when 
                attempting to learn the manifold structure of the data.
        
        
        Reference to the authors of UMAP: 
        McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, 
        ArXiv e-prints 1802.03426, 2018
        
        '''
        
        
        if n_comps == None:
            
            n_comps = n_neighbors
        
        assert n_comps <= n_neighbors, 'n_comps must be less than or equal to n_neighbors'
        
        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = umap.UMAP( n_neighbors = n_neighbors,  min_dist = 0.5, n_components = n_comps ).fit( x )
        
        self.umap_fitted = True
        
        return self.project_obj
        
        
            
    def transform_umap( self, x = None, return_data = False ):
        
        '''
        
        Project a dataset using the UMAP object fitted in the fit_umap() method: 
        
            - x: Array or dataframe, default = None, optional  
                 Contains the data to be transformed by the fitted umap object. If x is None then the data used to fit 
                 the umap object in fit_umap() will be used in transfrom_pca. 
                 
                 If x is not None the the fitted parameters of the umap object will be used to transform x. 
                 Furthermore, if a standardization method is specified and x is not None then the parameters of
                 the standardization object fitted in fit_umap() will be used to standardize x before proejcting 
                 the data.
            
            - return_data: bool, default = False
                 If True the function will return the projected dataset as a numpy array
        
        '''
        
        assert self.umap_fitted == True, 'No UMAP object has been fitted'
        
        
        if self.standard_method != None:
        
            if type( x ) != type( None ): 

                x = self.transform_standard( x, return_data = True )

            else: 

                x = self.standard_data
        
        
        self.project_data = self.project_obj.transform( x )
        
        if return_data == True: 
            return self.project_data
        
        
        
    def fit_transform_umap( self, x, n_comps = None, n_neighbors = 50, return_data = False ):
        
        '''
        
        Fit a UMAP object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the pca object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                Specifies the number of UMAP components to fit to the dataset
            
            - n_neighbors: int, default = 50
                This parameter controls how UMAP balances local versus global structure in the data. 
                It does this by constraining the size of the local neighborhood UMAP will look at when 
                attempting to learn the manifold structure of the data.
                
            - return_data: bool, default = False
                If True the function will return the projected dataset as a numpy array
        
        
        Reference to the authors of UMAP: 
        McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, 
        ArXiv e-prints 1802.03426, 2018
        
        '''
        
        if n_comps == None:
            
            n_comps = n_neighbors
        
        assert n_comps <= n_neighbors, 'n_comps must be less than or equal to n_neighbors'
        
        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = umap.UMAP( n_neighbors = n_neighbors,  min_dist = 0.5, n_components = n_comps ).fit( x )        
        
        self.project_data = self.project_obj.transform( x )
        
        self.umap_fitted = True
        
        if return_data == True: 
            return self.project_data
        
        

    def fit_transform_tsne( self, x, n_comps = None, return_data = False ):

        '''

        Fit a TSNE object to a dataset with the inputs: 

            - x: Array or dataframe 
                 Contains the data to be fitted by the pca object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 

            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                Specifies the number of UMAP components to fit to the dataset

            - n_neighbors: int, default = 50
                This parameter controls how UMAP balances local versus global structure in the data. 
                It does this by constraining the size of the local neighborhood UMAP will look at when 
                attempting to learn the manifold structure of the data.

            - return_data: bool, default = False
                If True the function will return the projected dataset as a numpy array


        Reference to the authors of UMAP: 
        McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, 
        ArXiv e-prints 1802.03426, 2018

        '''

        if n_comps == None:

            n_comps = 3

        assert n_comps <= 3, 'n_comps must be less than or equal to 3'

        if self.standard_method != None: 

            x = self.fit_transform_standard( x, return_data = True )

        self.project_obj = TSNE( n_components = n_comps, n_iter=1000 )       

        self.project_data = self.project_obj.fit_transform( x ) 

        self.tsne_fitted = True

        if return_data == True: 
            return self.project_data
        
        
    def fit_phate(self, x, n_comps = None, knn=5, decay=40, n_landmark=2000, t='auto', gamma=1, 
                  n_pca=100, mds_solver='sgd', knn_dist='euclidean', mds_dist='euclidean', 
                  mds='metric', n_jobs=1, random_state=123, verbose=1 ):
                
        '''
        
        Fit a PHATE object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the PHATE object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                Specifies the number of PHATE components to fit to the dataset
        
        
        Reference to the authors of PHATE: 
        Moon KR, van Dijk D, Zheng W, et al. (2017), PHATE: A Dimensionality Reduction Method for 
        Visualizing Trajectory Structures in High-Dimensional Biological Data, BioRxiv.
        
        Documetnation: https://phate.readthedocs.io/en/stable/api.html#id2
        
        '''
        
        if n_comps == None:
            
            n_comps = min( x.shape )

            n_comps = min((n_comps, n_landmark))
        
        
        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = phate.PHATE(n_components = n_comps, knn = knn, decay = decay, n_landmark = n_landmark, 
                                       t = t, gamma = gamma, n_pca = n_pca, mds_solver = mds_solver, knn_dist = knn_dist, 
                                       mds_dist = mds_dist, mds = mds, n_jobs = n_jobs, random_state = random_stat, 
                                       verbose = verbose).fit( x )
        
        self.phate_fitted = True
        
        return self.project_obj
        
        
            
    def transform_phate( self, x = None, return_data = False ):
        
        '''
        
        Project a dataset using the UMAP object fitted in the fit_umap() method: 
        
            - x: Array or dataframe, default = None, optional  
                 Contains the data to be transformed by the fitted umap object. If x is None then the data used to fit 
                 the umap object in fit_umap() will be used in transfrom_pca. 
                 
                 If x is not None the the fitted parameters of the umap object will be used to transform x. 
                 Furthermore, if a standardization method is specified and x is not None then the parameters of
                 the standardization object fitted in fit_umap() will be used to standardize x before proejcting 
                 the data.
            
            - return_data: bool, default = False
                 If True the function will return the projected dataset as a numpy array
        
        '''
        
        assert self.phate_fitted == True, 'No PHATE object has been fitted'
        
        
        if self.standard_method != None:
        
            if type( x ) != type( None ): 

                x = self.transform_standard( x, return_data = True )

            else: 

                x = self.standard_data
        
        
        self.project_data = self.project_obj.transform( x )
        
        if return_data == True: 
            return self.project_data
        
        
        
    def fit_transform_phate(self, x, n_comps = None, knn=5, decay=40, n_landmark=2000, t='auto', gamma=1, 
                            n_pca=100, mds_solver='sgd', knn_dist='euclidean', mds_dist='euclidean', 
                            mds='metric', n_jobs=1, random_state=123, verbose=1, return_data = False ):
        
        '''
        
        Fit a PHATE object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the PHATE object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                Specifies the number of PHATE components to fit to the dataset
                
            - return_data: bool, default = False
                If True the function will return the projected dataset as a numpy array
        
        
        Reference to the authors of PHATE: 
        Moon KR, van Dijk D, Zheng W, et al. (2017), PHATE: A Dimensionality Reduction Method for 
        Visualizing Trajectory Structures in High-Dimensional Biological Data, BioRxiv.
        
        Documetnation: https://phate.readthedocs.io/en/stable/api.html#id2
        
        '''
        
        if n_comps == None:
            
            n_comps = min( x.shape )

            n_comps = min((n_comps, n_landmark))
        

        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = phate.PHATE(n_components = n_comps, knn = knn, decay = decay, n_landmark = n_landmark, 
                                       t = t, gamma = gamma, n_pca = n_pca, mds_solver = mds_solver, knn_dist = knn_dist, 
                                       mds_dist = mds_dist, mds = mds, n_jobs = n_jobs, random_state = random_state, 
                                       verbose = verbose).fit( x )        
        
        self.project_data = self.project_obj.transform( x )
        
        self.phate_fitted = True
        
        if return_data == True: 
            return self.project_data
    
    
    def fit_isomap(self, x, n_comps = None, n_neighbors = 5 ):
                
        '''
        
        Fit a Isomap object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the Isomap object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
            
            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                Specifies the number of Isomap components to fit to the dataset
        
        
        Reference to the authors of PHATE: 
        R7f4d308f5054-1 Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric framework for nonlinear 
        dimensionality reduction. Science 290 (5500)
        
        '''
        
        if n_comps == None:
            
            n_comps = min( x.shape )
        
        
        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = Isomap(n_components = n_comps, n_neighbors = n_neighbors).fit( x )
        
        self.isomap_fitted = True
        
        return self.project_obj
        
        
            
    def transform_isomap( self, x = None, return_data = False ):
        
        '''
        
        Project a dataset using the UMAP object fitted in the fit_umap() method: 
        
            - x: Array or dataframe, default = None, optional  
                 Contains the data to be transformed by the fitted umap object. If x is None then the data used to fit 
                 the umap object in fit_umap() will be used in transfrom_pca. 
                 
                 If x is not None the the fitted parameters of the umap object will be used to transform x. 
                 Furthermore, if a standardization method is specified and x is not None then the parameters of
                 the standardization object fitted in fit_umap() will be used to standardize x before proejcting 
                 the data.
            
            - return_data: bool, default = False
                 If True the function will return the projected dataset as a numpy array
        
        '''
        
        assert self.isomap_fitted == True, 'No Isomap object has been fitted'
        
        
        if self.standard_method != None:
        
            if type( x ) != type( None ): 

                x = self.transform_standard( x, return_data = True )

            else: 

                x = self.standard_data
        
        
        self.project_data = self.project_obj.transform( x )
        
        if return_data == True: 
            return self.project_data
        
        
        
    def fit_transform_isomap(self, x, n_comps = None, n_neighbors = 5, return_data = False):
        
        '''
        
        Fit a PHATE object to a dataset with the inputs: 
        
        Fit a Isomap object to a dataset with the inputs: 
        
            - x: Array or dataframe 
                 Contains the data to be fitted by the Isomap object. If a standardization method is specified
                 then x is standardized prior to fitting the pca object. 
        
            - n_comps: int, defalut = None, n_comps <= min( n_features, n_observations, n_neighbors ) 
                        Specifies the number of Isomap components to fit to the dataset
            
            - return_data: bool, default = False
                            If True the function will return the projected dataset as a numpy array
        
        
        Reference to the authors of PHATE: 
        Moon KR, van Dijk D, Zheng W, et al. (2017), PHATE: A Dimensionality Reduction Method for 
        Visualizing Trajectory Structures in High-Dimensional Biological Data, BioRxiv.
        
        Documetnation: https://phate.readthedocs.io/en/stable/api.html#id2
        
        '''
        
        if n_comps == None:
            
            n_comps = min( x.shape )
        

        if self.standard_method != None: 
            
            x = self.fit_transform_standard( x, return_data = True )
        
        self.project_obj = Isomap(n_components = n_comps, n_neighbors = n_neighbors).fit( x )        
        
        self.project_data = self.project_obj.transform( x )
        
        self.isomap_fitted = True
        
        if return_data == True: 
            return self.project_data

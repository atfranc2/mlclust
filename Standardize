from sklearn import preprocessing

class Standard: 
    
    '''
    
    Class will standardize data using one of the following standard_method specifications:
        
        - 'standardize'
        
        - 'center'
        
        - 'min_max'
    
    '''
    
    
    
    from sklearn import preprocessing
    
    
    
    def __init__( self, standard_method = None ):
        
        self.standard_method = standard_method
        
        self.standard_fitted = False
        
        self.standard_data = None
        
        self.standard_obj = None

        
        
    def reset_params( self ):
        
        '''
        
        Will reset the parameters of the class to their default values
        
        '''
        
        self.standard_fitted = False
        
        return self
    
    

    def fit_standard( self, x ):
        
        '''
        
        Will standardize the data by the method standardization_method specified in the __init__ fuction 
        and return the standardized dataset.
        
        '''

        if self.standard_method == 'standardize':
            
            self.standard_obj = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( x )

            
        elif self.standard_method == 'center':
            
            self.standard_obj = preprocessing.StandardScaler( with_mean=True, with_std=False ).fit( x )

            
        elif self.standard_method == 'min_max': 
            
            self.standard_obj = preprocessing.MinMaxScaler().fit( x )
        
        self.standard_fitted = True
            
        return self
    
    
    
    def transform_standard( self, x, return_data = False ):
        
        '''
        
        Will transform the data using the standrization object created in the fit_standard() method
        
        '''        
        
        assert self.standard_fitted == True
        
    
        if self.standard_method == 'standardize':
            
            self.standard_data = self.standard_obj.transform( x )

            
        elif self.standard_method == 'center':
            
            self.standard_data= self.standard_obj.transform( x )

            
        elif self.standard_method == 'min_max': 
            
            self.standard_data = self.standard_obj.transform( x )
        
        
        if return_data == True:
            return self.standard_data
    
    
    
    def fit_transform_standard( self, x, return_data = False ):
        
        '''
        
        Fit a standardization object and transform x using the fitted standardization object
        
        '''
        
    
        if self.standard_method == 'standardize':
            
            self.standard_obj = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( x )
            
            self.standard_data = self.standard_obj.transform( x )

            
        elif self.standard_method == 'center':
            
            self.standard_obj = preprocessing.StandardScaler( with_mean=True, with_std=False ).fit( x )
            
            self.standard_data = self.standard_obj.transform( x )

            
        elif self.standard_method == 'min_max': 
            
            self.standard_obj = preprocessing.MinMaxScaler().fit( x )
            
            self.standard_data = self.standard_obj.transform( x )
        
        
        self.standard_fitted = True
        
            
        if return_data == True:
            return self.standard_data
        
        

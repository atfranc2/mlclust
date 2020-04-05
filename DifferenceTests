
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng
import scipy.stats as stats
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DiffTests: 
    
    # Allows arguments to be placed in a function as a list
    def place_args( self, func, args ):
        return func( *args )

    
    
    # Performs Welches F-Test for unequal variance
    def welch_f( self, *argv ):

        args = []

        for i in argv: 
            args.append(i)

        means = []

        counts = []

        variances = []

        w = []

        mean_primes = []

        for i in args:
            mean = i.mean()
            var = i.var()
            count = i.count()
            w_stat = count / var
            mean_prime = w_stat * mean

            means.append( mean )
            variances.append( var )
            counts.append( count )
            w.append( w_stat )
            mean_primes.append(mean_prime)


        w_stat = sum(w)
        mean_prime_stat = sum(mean_primes) / w_stat

        a = []
        b = []

        for i in range(0, len(w)):
            a_stat = w[i] * (( means[i] - mean_prime_stat )**2)

            b_stat = ( ( 1 - (w[i] / w_stat) )**2 ) / (counts[i] - 1)

            a.append( a_stat )
            b.append( b_stat )

        k = len(w)

        a_stat = sum(a) / (k-1)

        b_stat = sum(b)

        part1 = ( 2 * b_stat * (k - 2) )

        part2 = (k**2) - 1

        part3 = part1 / part2

        F = a_stat / ( 1 + part3 ) 

        df1 = k - 1

        df2 = ((k**2) - 1) / (3 * b_stat)

        p_value = stats.f.pdf(F, df1,df2)

        return [F, p_value]
    
    
    def ANOVA( self, df, target_col, group_col, pairwise_method = 'median', target_type = 'binary', group_alpha = 0.01, pairwise_alpha = 1.88e-6 ):
        
        assert pairwise_method in ['median', 'mean'], 'The only valid arguments for pairwise_method are median for \
        mann-whitney U test of median intensity between spectral features, or mean for t-test between spectral features.'
        
        df = df.copy(deep=True)
        
        df_cols = df.columns
        
        df['Target'] = target_col

        df['Group'] = group_col

        # Pull the unique group identifiers
        groups = df[ 'Group' ].unique()        

        # Will store data from each group
        slices = []

        # Appends group data to slices
        for group in groups: 

            slices.append( df[ 'Target' ][df[ 'Group' ] == group ] )

        # Extracts p-value for Levenes test of unequal variances 
        levene_p_val = self.place_args( stats.levene, slices )[1]

        # Based on results from Levene will perform different global F-Tests
        if levene_p_val < group_alpha: 

            print( 'Variance between groups is not equal' )

            F = self.place_args( self.welch_f, slices )[1]

            print(F)

        else: 

            print( 'Variance between groups is equal' )

            F = self.place_args( stats.f_oneway, slices )[1]

        if F < group_alpha: 

            print( 'A significant difference between groups exists. Now perfroming Tukey comparisons...' )

            mc = MultiComparison( df[ 'Target' ], df[ 'Group' ] )

            #Perfrom the Tukey HSD test
            mc_results = mc.tukeyhsd(alpha = group_alpha)
            
            print(mc_results)
            
            groups.sort()
            
            combos = np.array([list(i) for i in combinations(groups,2)])
            
            anova_info = [combos, mc_results.pvalues]
            
            sig_combos = anova_info[0][anova_info[1] < group_alpha]
            
            df = df.set_index('Group')
            
            df = df[df_cols.values]
            
            signif_info = []
            
            if pairwise_method == 'median': 
                
                for i in sig_combos:
                    # The first cluster
                    g1 = df.loc[i[0]]

                    # The second cluster
                    g2 = df.loc[i[1]]

                    p_values = np.array([stats.mannwhitneyu( g1[i], g2[i] ).pvalue for i in df_cols])

                    # Find where values are less than alpha
                    sig_vals = p_values < pairwise_alpha

                    # Subset dataframe on sig_vals
                    sig_columns = df_cols[sig_vals]

                    signif_info.append([list(i), list(sig_columns)])
                    
            elif pairwise_method == 'mean':
                
                for i in sig_combos:
                    # The first cluster
                    g1 = df.loc[i[0]]

                    # The second cluster
                    g2 = df.loc[i[1]]

                    # Find p-values from tests
                    p_values = stats.ttest_ind(g1, g2, equal_var = False)[1]

                    # Find where values are less than alpha
                    sig_vals = p_value < pairwise_alpha

                    # Subset dataframe on sig_vals
                    sig_columns = df_cols[sig_vals]

                    signif_info.append([list(i), list(sig_columns)])
                    
            
            #Return the clusters with significantly different means
            return signif_info

        else:
            print( 'A significant difference between groups does not exist. Now exiting function...' )
    
    
    
    def chi(self, target_col, group_col, alpha=0.01):

        # Cross Tabulate
        table = pd.crosstab(group_col, target_col)

        # Test of Assocation
        chi_square_info=stats.chi2_contingency(table)

        # P-Value for Test
        chi_pvalue = chi_square_info[1]

        # Significant difference?
        if chi_pvalue < alpha:
            print('A significant difference exists at an alpha of {}'.format(alpha))
            return table

        else:
            print('There is no significant difference between clusters')
            raise ValueError( 'P-value larger than alpha' )
            
    
    
    def fisher(self, df, contingency_table, group_col, pairwise_method = 'median', group_alpha = 0.01, pairwise_alpha = 1.88e-6):
        
        df = df.copy(deep=True)
        
        df_cols = df.columns

        df['Group'] = group_col
        
        df = df.set_index('Group')
        
        p_vals = []
        
        comb = combinations(contingency_table.index, 2)
        
        sig_combos = []
        
        for i in list(comb):
            pvalue = stats.fisher_exact(contingency_table.iloc[list(i)])[1]
            if pvalue < group_alpha:
                sig_combos.append(i)

        print('The cluster pairs that are significantly different are: ', list(sig_combos))
        
        signif_info = []

        if pairwise_method == 'median': 
                
            for i in sig_combos:
                # The first cluster
                g1 = df.loc[i[0]]

                # The second cluster
                g2 = df.loc[i[1]]

                p_values = np.array([stats.mannwhitneyu( g1[i], g2[i] ).pvalue for i in df_cols])

                # Find where values are less than alpha
                sig_vals = p_values < pairwise_alpha

                # Subset dataframe on sig_vals
                sig_columns = df_cols[sig_vals]

                signif_info.append([list(i), list(sig_columns)])
                
        elif pairwise_method == 'mean':
            
            for i in sig_combos:
                # The first cluster
                g1 = df.loc[i[0]]

                # The second cluster
                g2 = df.loc[i[1]]

                # Find p-values from tests
                p_values = stats.ttest_ind(g1, g2, equal_var = False)[1]

                # Find where values are less than alpha
                sig_vals = p_values < pairwise_alpha

                # Subset dataframe on sig_vals
                sig_columns = df_cols[sig_vals]

                signif_info.append([list(i), list(sig_columns)])

        return signif_info
    
    
    def plot_fisher_groups(self, df, sig_combos, var_cols, index_name = 'Group', alpha = 0.01):
        
        col_map = pd.DataFrame( [[i for i in range(0, len( var_cols ))]], columns = var_cols )
        
        df = df.copy(deep=True)
        
        df = df.set_index(index_name)
        
        for i in sig_combos: 

            ids = col_map[ i[1] ].values

            ids = np.sort(ids)[0]

            start = 0 

            intervals = []

            for k in range(0, len( ids ) - 1 ): 

                diff = ids[k+1] - ids[k]

                if diff > 1: 

                    intervals.append( list( ids[start:k+1] ) )

                    start = k+1

                if k == len( ids ) - 2  :

                    intervals.append( list( ids[start:k+2] ) )

            y1 = df.loc[i[0][0]].median()
            y2 = df.loc[i[0][1]].median()
            
            plt.figure(figsize=(12,7))

            plt.plot( [ k for k in range (0, len( var_cols ) ) ], y1, color='purple' )
            plt.plot( [ k for k in range (0, len( var_cols ) ) ], y2, alpha = 0.8, color='cyan' )

            plt.title('Spectral variables that are significantly different at alpha={} in clusters:{}'.format(alpha, i[0]))

            plt.legend(i[0])

            for l in intervals: 

                plt.axvspan(min(l), max(l), color='red', alpha=0.5)

            plt.show()
    

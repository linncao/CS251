'''analysis.py
Run statistical analyses and plot Numpy ndarray data
LINN CAO NGUYEN PHUONG
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})


    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data


    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return self.data.select_data(headers, rows).min(axis = 0)


    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return self.data.select_data(headers, rows).max(axis = 0)


    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return self.min(headers, rows), self.max(headers, rows)


    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''
        subset = self.data.select_data(headers, rows)
        return subset.sum(axis = 0)/subset.shape[0]


    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''
        subset = self.data.select_data(headers, rows)
        subset_square = subset**2
        return (subset_square.sum(axis = 0) - 2*self.mean(headers, rows)*subset.sum(axis = 0) + subset.shape[0]*(self.mean(headers, rows)**2))/(subset.shape[0] - 1)


    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return self.var(headers, rows)**0.5
    
    # EXTENSION
    def mode(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Mode values for each of the selected header variables
        '''
        vals, counts = np.unique(self.data.select_data(headers, rows), return_counts=True)
        index = np.argmax(counts)
        return vals[index]
    
    # EXTENSION
    def median(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Median values for each of the selected header variables
        '''
        x = np.array(self.data.select_data(headers, rows))
        xlen = np.prod(x.shape)
        np.sort(x)

        if xlen % 2 == 0:
            median1 = x[xlen//2]
            median2 = x[xlen//2 - 1]
            median = (median1 + median2)/2
        else:
            median = x[xlen//2]
            
        return median
    
    # EXTENSION
    def rsquared(self, array1, array2):
        '''Computes the R squared value for 2 header variables in the data object.

        Parameters:
        -----------
        array1: ndarray
        array2: ndarray

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            R squared value for the 2 selected header variables
        '''
        correlation_matrix = np.corrcoef(array1, array2)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        return r_squared
        

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()


    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        x = self.data.select_data([ind_var])
        x = x.reshape(x.shape[0])
        y = self.data.select_data([dep_var])
        y = y.reshape(y.shape[0])

        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        
        return x, y
    
    # EXTENSION
    def hist2d(self, ind_var, dep_var, bins, title):
        '''Creates a simple 2D histogram with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        x = np.array(self.data.select_data([ind_var]))
        x = x.reshape(x.shape[0])
        y = np.array(self.data.select_data([dep_var]))
        y = y.reshape(y.shape[0])
        
        plt.hist2d(x, y, bins)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.colorbar()
        
        return x, y
        

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize = fig_sz, sharex = False, sharey = False)
        for row in range(axes.shape[0]):
            x = self.data.data[:, self.data.header2col[data_vars[row]]]
            for col in range(axes.shape[0]):
                y = self.data.data[:, self.data.header2col[data_vars[col]]]
                axes[col][row].scatter(x, y, marker = 'o')
                if col != (axes.shape[0] - 1):
                    axes[col][row].tick_params(labelbottom = False)
                if row != 0:
                    axes[col][row].tick_params(labelleft = False)

        for i in range(len(data_vars)):
            plt.setp(axes[-1, i], xlabel = data_vars[i])
            plt.setp(axes[i, 0], ylabel = data_vars[i])

        fig.suptitle(title)
        
        return fig, axes
    
            
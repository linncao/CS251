'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        super().__init__(data)
        self.orig_dataset = orig_dataset


    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        projected = self.orig_dataset.select_data(headers)
        header2col = {}
        for i in range(len(headers)):
            header2col[headers[i]] = i
        self.data = data.Data(filepath=None, headers=headers, data=projected,header2col=header2col)


    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        Dh = np.hstack((self.data.data, np.ones((self.data.data.shape[0], 1))))
        return Dh


    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        num_proj_vars = self.get_data_homogeneous().shape[1]
        t = np.eye(num_proj_vars)
        for i in range(num_proj_vars - 1):
            t[i, num_proj_vars - 1] = magnitudes[i]
        return t


    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        num_proj_vars = self.get_data_homogeneous().shape[1]
        s = np.eye(num_proj_vars)
        for i in range(num_proj_vars - 1):
            s[i, i] = magnitudes[i]
        return s
        

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        result = (self.translation_matrix(magnitudes)@self.get_data_homogeneous().T).T
        self.data = data.Data(data=result[:,:-1].copy(), headers=self.data.get_headers(), header2col = self.data.get_mappings())
        return self.data.data


    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        result = (self.scale_matrix(magnitudes)@self.get_data_homogeneous().T).T
        self.data = data.Data(data=result[:,:-1].copy(), headers=self.data.get_headers(), header2col = self.data.get_mappings())
        return self.data.data


    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        result = (C@self.get_data_homogeneous().T).T
        self.data = data.Data(data=result[:,:-1].copy(), headers=self.data.get_headers(), header2col = self.data.get_mappings())
        return self.data.data


    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        global_min = self.data.data.min()
        global_max = self.data.data.max()
        global_range = global_max - global_min
        T = self.translation_matrix([-global_min]*len(self.data.get_headers()))
        S = self.scale_matrix([1/global_range]*len(self.data.get_headers()))
        result = S@T
        self.transform(result)
        return self.data.data


    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        T = self.translation_matrix(-np.min(self.data.data,axis = 0))
        S = self.scale_matrix(1/(np.max(self.data.data,axis = 0)-np.min(self.data.data,axis = 0)))
        result = S@T
        self.transform(result)
        return self.data.data


    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        R = np.eye(4)
        rad = np.deg2rad(degrees)
        if header == self.data.get_headers()[0]:
            R[1, 1] = np.cos(rad)
            R[1, 2] = -np.sin(rad)
            R[2, 1] = np.sin(rad)
            R[2, 2] = np.cos(rad)
        elif header == self.data.get_headers()[1]:
            R[0, 0] = np.cos(rad)
            R[0, 2] = np.sin(rad)
            R[2, 0] = -np.sin(rad)
            R[2, 2] = np.cos(rad)
        else:
            R[0, 0] = np.cos(rad)
            R[0, 1] = -np.sin(rad)
            R[1, 0] = np.sin(rad)
            R[1, 1] = np.cos(rad)
        return R


    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        R = self.rotation_matrix_3d(header, degrees)
        result = (R@self.get_data_homogeneous().T).T
        self.data = data.Data(data=result[:,:-1].copy(), headers=self.data.get_headers(), header2col = self.data.get_mappings())
        return self.data.data
    
    # EXTENSION 3.1
    def rotation_matrix_2d(self, header, degrees):
        '''Make an 2-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable which the projected dataset should be rotated from. 
            If it's the first variable of the 2, the rotation is clockwise. Else, the rotation is counterclockwise.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(3, 3). The 2D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        R = np.eye(3)
        rad = np.deg2rad(degrees)
        if header == self.data.get_headers()[0]:
            R[0, 0] = np.cos(rad)
            R[0, 1] = -np.sin(rad)
            R[1, 0] = np.sin(rad)
            R[1, 1] = np.cos(rad)
        else:
            R[0, 0] = np.cos(rad)
            R[0, 1] = np.sin(rad)
            R[1, 0] = -np.sin(rad)
            R[1, 1] = np.cos(rad)
        return R
    
    # EXTENSION 3.2
    def rotate_2d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable which the projected dataset should be rotated from. 
            If it's the first variable of the 2, the rotation is clockwise. Else, the rotation is counterclockwise.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        R = self.rotation_matrix_2d(header, degrees)
        result = (R@self.get_data_homogeneous().T).T
        self.data = data.Data(data=result[:,:-1].copy(), headers=self.data.get_headers(), header2col = self.data.get_mappings())
        return self.data.data


    def scatter3d(self, xlim, ylim, zlim, better_view=False):
        '''Creates a 3D scatter plot to visualize data the x, y, and z axes are drawn, but not ticks

        Axis labels are placed next to the POSITIVE direction of each axis.

        Parameters:
        -----------
        xlim: List or tuple indicating the x axis limits. Format: (low, high)
        ylim: List or tuple indicating the y axis limits. Format: (low, high)
        zlim: List or tuple indicating the z axis limits. Format: (low, high)
        better_view: boolean. Change the view so that the Z axis is coming "out"
        '''
        if len(self.data.get_headers()) != 3:
            print("need 3 headers to make a 3d scatter plot")
            return

        headers = self.data.get_headers()
        xyz = self.data.get_all_data()

        if better_view:
            # by default, matplot lib puts the 3rd axis heading up
            # and the second axis heading back.
            # rotate it so that the second axis is up and the third is forward
            R = np.eye(3)
            R[1, 1] = np.cos(np.pi/2)
            R[1, 2] = -np.sin(np.pi/2)
            R[2, 1] = np.sin(np.pi/2)
            R[2, 2] = np.cos(np.pi/2)
            xyz = (R @ xyz.T).T

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Scatter plot of data in 3D
        ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.plot(xlim, [0, 0], [0, 0], 'k')
        ax.plot([0, 0], ylim, [0, 0], 'k')
        ax.plot([0, 0], [0, 0], zlim, 'k')
        ax.text(xlim[1], 0, 0, headers[0])

        if better_view:
            ax.text( 0, zlim[0], 0, headers[2])
            ax.text( 0, 0, ylim[1], headers[1])
        else:
            ax.text(0, ylim[1], 0, headers[1])
            ax.text(0, 0, zlim[1], headers[2])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        plt.show()
        
    # EXTENSION 1
    def scatter4d(self, var1, var2, var3, var4, figsize, title):
        '''Creates a 3D scatter plot to visualize 4 variables of data

        Parameters:
        -----------
        var1: str.
            Name of variable that is plotted along the x axis
        var2: str.
            Name of variable that is plotted along the y axis
        var3: str.
            Name of variable that is plotted along the z axis
        var4: str.
            Name of variable that is depicted using the colorbar
        figsize: list.
            Size of the plot.
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot
        z. ndarray. shape=(num_data_samps,)
            The z values that appear in the scatter plot
        c. ndarray. shape=(num_data_samps,)
            The c values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array(self.data.select_data([var1]))
        y = np.array(self.data.select_data([var2]))
        z = np.array(self.data.select_data([var3]))
        c = np.array(self.data.select_data([var4]))
        
        img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel(var3)
        cbar = fig.colorbar(img)
        cbar.set_label(var4)
        plt.title(title)
        
        return x, y, z, c
    
    
    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        plt.style.use('ggplot')
        x = np.array(self.orig_dataset.select_data([ind_var]))
        x = x.reshape(x.shape[0])
        y = np.array(self.orig_dataset.select_data([dep_var]))
        y = y.reshape(y.shape[0])
        z = np.array(self.orig_dataset.select_data([c_var]))
        z = z.reshape(y.shape[0])
        
        color_map = palettable.colorbrewer.sequential.Greys_9
        plt.scatter(x, y, c=z, s=75, cmap=color_map.mpl_colormap, edgecolor='black')
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.colorbar(label = c_var)


    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax

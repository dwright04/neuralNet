import numpy as np

class NeuralNetwork(object):
    """
    A simple feed forward neural network.
        
    The NeuralNet should behave as if it inherited from sklearn.base.BaseEstimator,
    but without being dependent on sklearn.  This NeuralNet has been designed to
    interact with sklearn grid_search.GridSearch or any functionality of the
    cross_validation module.
    
    Attributes
    ----------
    
    architecture : dict of int keys and int values, optional (default={1:25})
        Defines the hidden layers (keys) and number of neurons (values).

    LAMBDA : float, optional (default=0.0)
        Regularization parameter.
        
    neuron : string, optional (default='sigmoid')
        Specifies the type of neuron to use.  It must be one of 'sigmoid' or 'tanh'.
        
    optimiser : string, optional (default='fmin_cg')
        Algorithm to use for optimisation of neural network parameters.  Must be one 
        of 'grad_decent', 'stoch_grad_decent' or 'fmin_cg'.
    
    maxiter : int, optional (default=1000)
        Maximum number of iterations for the optimisation algorithm.
        
    Methods
    -------
    fit(X, y)
        Fit the neural network model according to the given training data.
    
    get_params([deep])
        Get parameters for this estimator.
        
    predict(X)
        Perform classification on samples in X.
        
    set_params(**params)
        Set the parameters of this estimator.
    
    Notes
    -----
    
    More neuron types are likely to be added in the future.
    More optimisation functions are likely to be added in future.
    """

    def __init__(self, architecture={1:25}, LAMBDA=0.0, neuron='sigmoid', optimiser='fmin_cg', maxiter=1000):
        ### Do some house keeping ###
        from scipy import optimize
        scipy_optimisers = {'fmin_cg':optimize.fmin_cg, \
                            'fmin_bfgs':optimize.fmin_bfgs}
        self.architecture = architecture
        self.LAMBDA = LAMBDA
        self.neuron = neuron
        self.maxiter = maxiter
        # setup the optimiser
        try:
          self.optimiser = scipy_optimisers[optimiser]
        except:
            raise NotImplementedError
            # TODO : gradient descent
        # private variable to store trained parameters of the network.
        self._trainedParams = None
    
    def __repr__(self):
        return "%r" % self.__class__

    def fit(self, X, y):
        """
        Fit the Neural Network model according to the training data.
        
        Parameters
        ----------
        X : nd_array, shape(m, n)
            Training vectors , where m is the number of training samples and n is
            the number of features.
            
        y : nd_array, shape(m,)
            Target values (class lables for classification, real numbers for regression).
                
        Returns
        -------
        self : object
            Returns self.
        """
        # remove any padded dimensions.
        X = np.squeeze(X)
        y = np.squeeze(y)
        # get the number of training examples and number of features.
        m, n = np.shape(X)
        # set up the architecture of the neural network accordingly.
        # size input layer = number of features.
        self.architecture[0] = n
        if len(np.unique(y)) > 2:
            # if number of labels > 2 set the size of output layer to number unique labels.
            self.architecture[len(list(self.architecture.keys()))] = len(np.unique(y))
        elif len(np.unique(y)) == 2:
            # if number of labels = 2 set the size of output layer to 1.
            self.architecture[len(list(self.architecture.keys()))] = 1

        ### Get the right neurons ###
        if self.neuron == 'sigmoid':
            from Neuron import SigmoidNeuron
            neuron = SigmoidNeuron.fire
            dneuron = SigmoidNeuron.dfire
        elif self.neuron == 'tanh':
            from Neuron import TanhNeuron
            neuron = TanhNeuron.fire
            dneuron = TanhNeuron.dfire

        ### Define f and f_prime for optimizer ###
        def costFunction(params, *args):
            input, targets = args
            cost, grad = self.costFunction(params, input, targets)
            return cost

        def costFunctionGradient(params, *args):
            input, targets = args
            cost, grad = self.costFunction(params, input, targets)
            return grad
        
        print(("[*] Training %s" % (self)))

        try:
            y = oneHotEncoding(y)
        except AttributeError:
            y = y

        args = (X, y)
        initialParams = self.initialise()
        params = optimiser(costFunction, x0=initialParams, fprime=costFunctionGradient, \
                                  args=args, maxiter=self.maxiter)
            
        self._trainedParams = params
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def get_params(self, deep=True):
        return {'architecture' : self.architecture,\
                'LAMBDA': self.LAMBDA,\
                'neuron': self.neuron,\
                'optimiser': self.optimiser,\
                'maxiter': self.maxiter}

    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            self.setattr(parameter, value)

    def setattr(self, parameter, value):
        """
        """
        if parameter in self.__dict__:
            self.__dict__.__setitem__(parameter, value)
        else:
            raise AttributeError(parameter)

    def randInitParams(self, nIn, nOut):
        """
        """
        # Note: The first row corresponds to the parameters for the bias units
        r = np.sqrt(6) / (np.sqrt(nOut + nIn + 1))
        return np.random.rand(nOut, (nIn + 1)) * 2 * r - r

    def initialise(self):
        
        """
        """
        # initialise initialParams by initialising the weights for the connection between the input and
        # first hidden layer.
        initialParams = np.ravel(self.randInitParams(self.architecture[0], \
                                                     self.architecture[1]), order="F")
        # loop through the remaining layers and intialise the weights
        for layer in list(self._architecture.keys())[1:-1]:
            initialParams = np.concatenate((initialParams, \
                            np.ravel(self.randInitParams(self.architecture[layer], \
                                                        self.architecture[layer+1]), \
                                                        order="F")), axis=0)
        return initialParams

    def reshapeParams(self, params):
        """
        """
        
        thetas = {1:np.reshape(params[0:self.architecture[1] * (self.architecture[0] + 1)], \
                              (self.architecture[1], (self.architecture[0] + 1)), order="F")}
        # index of the last weight for the first layer in the vector
        lastIndex = self.architecture[1] * (self.architecture[0] + 1)
        for layer in list(self.architecture.keys())[2:]:
            thetas[layer] = np.reshape(params[lastIndex:lastIndex + \
                                      (self.architecture[layer] * (self.architecture[layer-1] + 1))], \
                                      (self.architecture[layer], (self.architecture[layer-1] + 1)), order="F")
            lastIndex = lastIndex + (self.architecture[layer] * (self.architecture[layer-1] + 1))
        return thetas

    def oneHotEncoding(self, y):
        """
        The indicator function maps the targets for a classification problem from
        integers to vectors of 1's and 0's.  For a 3 class problem the targets are
        mapped as shown below:
        
        target      indicator function
        1             [1, 0, 0]
        2             [0, 1, 0]
        3             [0, 0, 1]
        
        This implementation assumes that labels are indexed from 1 not 0.
        
        parameters:
        
        targets: <numpy-array> 1xm vector of discrete class labels/targets.
        
        returns:
        
        indicatorFunction: <numpy-array> kxm array, where k is the number of classes
        and m the number of training examples.  The indicator function
        produces a kx1 vector denoting which class a given training example
        belongs to.
        """
        # get the number of discrete classes
        numClasses = len(np.unique(targets))
        # intialise the indicator function to an array fo zeros
        indicatorFunction = np.zeros((np.shape(targets)[0], numClasses))
        
        # loop through each training example and produce the indicator function
        for i in range(np.shape(targets)[0]):
            # vector of 0's except for the index corresponding to the target
            indicatorFunction[i,int(targets[i])] = 1
        # transpose so correct for comparison with hypothesis
        return indicatorFunction.transpose()

    def feedForward(self, thetas, input, regTerm, m, neuron):
        """
        """
        # setup some variables for the calculation
        activs = {1:input} # activation of the input layer is the input
        ### Forward Propagation ###
        # calculate the mapping of the input between all layers except the output layer.
        # While doing this calculate the regularisation term to avoid looping through layers
        # a second time.
        for layer in list(thetas.keys())[:-1]:
            z = np.dot(thetas[layer], activs[layer])
            activs[layer+1] = np.concatenate((np.tile(1, (1, m)), neuron(z)), axis=0) # add bias unit
            regTerm += np.sum(np.multiply(thetas[layer][:,1:], thetas[layer][:,1:]))
        regTerm += np.sum(np.multiply(thetas[list(thetas.keys())[-1]][:,1:], thetas[list(thetas.keys())[-1]][:,1:]))
        # calculate the activation of the output layer also known as the hypothesis
        z = np.dot(thetas[list(thetas.keys())[-1]], activs[list(activs.keys())[-1]])
        hypothesis = neuron(z)
        # checks for numerical instabilities
        if 1 in hypothesis:
            # np.log(1-1) = np.log(0) = -inf ( divide by zero encountered in log)
            # subtract off small number from 1
            hypothesis[np.where(hypothesis == 1)] = hypothesis[np.where(hypothesis == 1)] - 1e-9
        if 0 in hypothesis:
            # np.log(0) = -inf ( divide by zero encountered in log)
            hypothesis[np.where(hypothesis == 0)] = hypothesis[np.where(hypothesis == 0)] + 1e-9
        
        return hypothesis, activs, regTerm

    def backProp(self, thetas, hypothesis, activs, targets, m, dneuron):
        """
        """
        deltas = {} # dictionary to store errors for each layer during back prop
        grads = {} # dictionary to store gradients for each layer during back prop
        ### Back Propagation ###
        numLayers = len(list(self.architecture.keys()))
        deltas[numLayers] = np.subtract(hypothesis, targets)
        for layer in range(numLayers, 2, -1):
            deltas[layer-1] = np.multiply(np.dot(thetas[layer-1].transpose(), deltas[layer]), \
                                          dneuron(activs[layer-1]))
            deltas[layer-1] = deltas[layer-1][1:,:]
        for layer in list(thetas.keys()):
            grad = 1/m * (np.dot(deltas[layer+1], activs[layer].transpose()))
            grad[:,1:] = grad[:,1:] + 1/m * (self.LAMBDA * thetas[layer][:,1:])
            grads[layer] = grad
        gradients = np.ravel(grads[1], order="F")
        for layer in list(grads.keys())[1:]:
            gradients = np.concatenate((gradients, np.ravel(grads[layer], order="F")), axis=0)
        return gradients

    def costFunction(self, params, X, y, m, neuron, dneuron):
        # setup some variables for the calculation
        thetas = self.reshapeParams(params) # reshape the vecotr into matrices
        m = float(m)
        regTerm = 0 # varaible to accumulate regularisation terms
    
        ### Feed the inputs forward though the network
        hypothesis, activations, regTerm = self.feedForward(thetas, input, regTerm, m, neuron)
    
        ### Cost Function Calculation ###
        # add the last theta term to the regularisation calulation
        cost = np.sum(np.multiply(-y, np.log(hypothesis)) - \
                      np.multiply((1-y), (np.log(1-hypothesis))))
                  
        cost = 1/m * (cost + (self.LAMBDA*0.5*regTerm))
                  
        ### Backpropagate errors though network ###
        gradients = self.backProp(thetas, hypothesis, activations, targets, m, dneuron)
                  
        return cost, gradients

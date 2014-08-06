import numpy as np

class Neuron(object):
    """
    Neuron to be used in conjunction with NeuralNet objects.
        
    This is an abstract class whose methods should be overridden by
    subclasses defining the behaviour of specific neuron types.
    
    Methods
    -------
    fire(stimuli)
        calculate the firing of the neuron given the stimuli.
            
    dfire(stimuli)
        calculate the derivative of the neuron fire function
        given the input stimuli.
    """

    def fire(self, stimuli):
        """
        Activation function for a neuron.
        
        Must be overriden by subclasses.
        
        Parameters
        ----------
        stimuli : float or nd_array
            Input to a neuron or a layer of neurons, where each element
            of an array corresponds to an individual neuron.
        """
        raise NotImplementedError

    def dfire(self, stimuli):
        """
        Derivative of the activation function defined in fire method.

        Must be overriden by subclasses.
        
        Parameters
        ----------
        stimuli : float or nd_array
            Input to a neuron or a layer of neurons, where each element
            of an array corresponds to an individual neuron.
        """
        raise NotImplementedError

    def __repr__(self):
        return "%r" % self.__class__

class SigmoidNeuron(Neuron):
    """
    Neuron object to be used in conjunction with NeuralNet objects.
        
    This is a concrete implementation of the Neuron abstract class
    for a sigmoid neuron.
    
    Methods
    -------
    fire(stimuli)
        calculate the firing of the neuron given the stimuli.
        
    dfire(stimuli)
        calculate the derivative of the neuron fire function
        given the input stimuli.
    """
    def fire(self, stimuli):
        """
        Activation of a sigmoid function.
        
        Parameters
        ----------
        stimuli : float or nd_array
            Input to a neuron or a layer of neurons, where each element
            of an array corresponds to an individual neuron.
            
        Returns
        -------
        activation : float or nd_array
            The result of calculating the sigmoid function on the stimuli.
        """
        return (1 / (1 + np.exp(-stimuli)))

    def dfire(self, stimuli):
        """
        Derivative of the sigmoid activation function defined in fire method.
        
        Parameters
        ----------
        stimuli : float or nd_array
            Input to a neuron or a layer of neurons, where each element
            of an array corresponds to an individual neuron.
            
        Returns
        -------
        dactivation : float or nd_array
            The result of calculating the derivative of the sigmoid
            function on the stimuli.
        """
        return np.multiply(self.fire(stimuli), (1 - self.fire(stimuli)))

class TanhNeuron(Neuron):
    """
    Neuron object to be used in conjunction with NeuralNet objects.
        
    This is a concrete implementation of the Neuron abstract class
    for a tanh neuron.
        
    Methods
    -------
    fire(stimuli)
        calculate the firing of the neuron given the stimuli.
        
    dfire(stimuli)
        calculate the derivative of the neuron fire function
        given the input stimuli.
    """
    def fire(self, stimuli):
        """
        Activation of a tanh function.
            
        Parameters
        ----------
        stimuli : float or nd_array
            Input to a neuron or a layer of neurons, where each element
            of an array corresponds to an individual neuron.
            
            
        Returns
        -------
        activation : float or nd_array
            The result of calculating the tanh function on the stimuli.
        """
        return np.tanh(stimuli)
    
    def dfire(self, stimuli):
        """
        Derivative of the tanh activation function defined in fire method.
            
        Parameters
        ----------
        stimuli : float or nd_array
            Input to a neuron or a layer of neurons, where each element
            of an array corresponds to an individual neuron.
            
        Returns
        -------
        dactivation : float or nd_array
            The result of calculating the derivative of the tanh function
            on the stimuli.
        """
        return np.multiply(self.fire(stimuli),\
                           np.multiply((1 - self.fire(stimuli)),\
                                       (1 - self.fire(stimuli))))

from pico import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

# A simple barebones implementation of Dense Neural Networks.


class NN:
    '''Good old densely connected neural network.'''
    
    def __init__(
            self, 
            input_size : int, output_size : int, 
            hiddens : list[int] = None, init : list = None
        ) -> None:
        """Creates a new network.

        Args:
            input_size (int): Number of inputs
            output_size (int): Number of outputs
            hiddens (list[int], optional): Number of cells in hidden layers. Defaults to [16, 16].
        """
        
        hiddens = [16, 16] if hiddens is None else hiddens
        layers = hiddens + [output_size]
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Weights and biases in each layer.
        self.layers : list[tuple[tensor, tensor]] = [] if init is None else init
        prev_size = input_size
        
        #If layers are not provided, initialize here:
        if init is None:
            for _, each in enumerate(layers):
                self.layers.append((
                    glorot_uniform(each, prev_size, in_size=prev_size, out_size=each).track(), # weights
                    glorot_uniform(each, 1, in_size=prev_size, out_size=each).track()  # biases
                ))
                prev_size = each
    
    
    def __call__(self, x : tensor) -> tensor:
        '''
        Forward-passes `x` through the layers. `x` must be `(batch size, input size, 1)` shaped.
        The output will be `(batch size, output size, 1)` shaped.
        '''
        batch_size = x.shape[0]
        
        for i, (w, b) in enumerate(self.layers):
            w = tile(w, batch_size)
            b = tile(b, batch_size)
            x = (w @ x) + b # Dense layer
            if i < (len(self.layers) - 1): 
                x = relu(x) #activation for all layers except last
        
        # activation for last layer, if any, goes here:
        # x = sigmoid(x)
        
        return x
    
    @property
    def size(self) -> int:
        '''Returns the number of scalar parameters in the entire network.'''
        s = 0
        for w, b in self.layers:
            s += w.size + b.size
        return s
    
    def save(self, file) -> None:
        '''
        Saves the network and its weights to a writeable buffer `file` using pickle. 
        Example:
        ```
        with open('./jarvis.model', 'wb') as file:
            jarvis.save(file)
        ```
        '''
        pickle.dump({
            'layers' : self.layers,
            'input_size' : self.input_size,
            'output_size' : self.output_size,
        }, file)
    
    def load(file) -> object:
        '''
        Loads and returns a network and its weights from a readable buffer `file` using pickle. 
        Example: 
        ```
        with open('./jarvis.model', 'rb') as file:
            jarvis = NN.load(file)
        ```
        '''
        data = pickle.load(file)
        isize = data['input_size']
        osize = data['output_size']
        return NN(isize, osize, init=data['layers'])




def train_model(
        model : NN, X : tensor, Y : tensor,
        batch_size : int = 64, alpha : float = 1e-1, beta = 1e3,
        epochs : int = 1
    ) -> list[float]:
    '''
    This is an example of how a typical training loop would be implemented using `pico.tensor`s.
    This particular example uses a simple optimizer, which can be implemented in many different ways.
    Returns history of losses while training.
    '''
    pointers = '⣾⣽⣻⢿⡿⣟⣯⣷'

    losses = []
    
    # full batch size
    length = X.shape[0]
    

    for each in range(epochs):
        # Learning rate depends on previous 2 losses; given by  alpha/(|losses[-1] - losses[-2]| * 1000).
        # It has a hard cap at 10alpha.
        # It is not fully stable, but works good enough.
        if each <= 1: _alpha = alpha
        else:
            p, pp = losses[-1], losses[-2]
            _alpha = min(alpha / (max(np.abs(pp - p), 1e-10) * beta), 10 * alpha)
        
        # getting gradients and updating:
        for i in range((length // batch_size)):
            
            ycap = model(X[i * batch_size : (i + 1) * batch_size])
            ytrue = Y[i * batch_size : (i + 1) * batch_size]
            
            # MSE loss
            loss = ((ycap - ytrue) / (batch_size ** 0.5)) ** 2
            
            loss = squeeze(loss)
            loss = sum(sum(loss), 0)
            loss = squeeze(squeeze(loss))
            
            # Updating weights if loss is not NaN:
            if not np.isnan(loss.value).any():
                loss.reverse()
                for w, b in model.layers:
                    w.value -= (_alpha * w.grad)
                    b.value -= (_alpha * b.grad)
                loss.clear_grads()

        # Evalution loss, evaluated on entire training X.
        eval_loss = ((model(X) - Y) / (length ** 0.5)) ** 2
        eval_loss = sum(sum(squeeze(eval_loss)), 0)
        eval_loss = eval_loss.item
        print(f'\rEpoch {each+1:4}/{epochs}: {eval_loss:.6f} {pointers[each % len(pointers)]} ({_alpha = :2.6f})', end='')
        losses.append(eval_loss)
    
    return losses



if __name__ == '__main__':
    
    print('Sample NN training.')
    
    # Simple curve fitting.
    # Defining the curve:
    f = lambda x: (x - 0.743) * (x - 0.097) * (x + 0.787)
    
    # Making dataset
    n = 100
    x = np.linspace(-1, 1, n)
    y = f(x)
    
    # shuffling the dataset for better results
    trainx = x.copy()
    np.random.shuffle(trainx)
    trainy = f(trainx)
    
    #making the model:
    sine_model = NN(input_size=1, output_size=1, hiddens=[32, 32])
    print(f"Model has {sine_model.size} params.")
    
    # training:
    hist = train_model(
            sine_model, 
            trainx.reshape((n, 1, 1)), trainy.reshape((n, 1, 1)), 
            epochs = 1000, alpha= 1
        )
    
    # prediction:
    ypred = sine_model(x.reshape((n, 1, 1))).value.reshape((n,))
    
    #plotting:
    fig, ax = plt.subplots(2, 1, figsize = (10, 8))
    ax[0].plot(x, y, 'g.', label = 'Data', alpha = 0.5)
    ax[0].plot(x, ypred, 'b.', label = 'Model', alpha = 0.5)
    ax[0].legend()
    ax[1].plot(hist)
    plt.show()
    
    
#IN PYTHON3
import numpy as np
from load_dataset import mnist
import matplotlib.pyplot as plt

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    x = cache
    dx = np.where(x > 0, dA, 0)
    return dx



def softmax_fc(x):

    #computes softmax probablities of forward fully connected layer
    # solving overflow problem
    if x.ndim == 1:
        x -= np.min(x)  
        x = np.exp(x)
        x /= np.sum(x)
    else:
        x -= np.min(x, axis=1, keepdims=True)  # solving overflow problem
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
       
    return x


def softmax_loss_backwards(out,y):
    eout = np.exp(out, dtype=np.float)
    probs = eout/np.sum(eout)

    p = np.sum(y*probs)
    cost = -np.log(p)   ## (Only data loss. No regularised loss)
    return cost

def zero_pad(X, pad):

    padded_x = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values = (0,0))
    return padded_x

def initialize_parameters(filters,filter_size):
    
    W=np.random.rand(filter_size,filter_size,1,filters)
    b=np.random.rand(1,1,1,filters)

    return W,b

def conv_single_forward(A, W, b):


    #multiplying one stride 
    
    s = A * W
    Z = np.sum(s)
    Z = np.float(Z + b)

    return Z


def conv_multi_forward(A_prev, W, b,stride,pad):
   
    (m, h_prev, w_prev, n_c_prev) = np.shape(A_prev)
    
    (f, f, n_c_prev, n_C) = np.shape(W)
    
    
    
    n_H = int((h_prev - f + 2 * pad) / stride) + 1
    n_W = int((w_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)

    Z = np.zeros((m, n_H, n_W, n_C))

    
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):                                 
        a_prev_pad = A_prev_pad[i,:,:,:]                
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                  
                    
                    start_verticle = h * stride
                    end_verticle = h * stride+ f
                    start_horizontal = w * stride
                    end_horizontal = w * stride + f
                    
                    a_slice_prev = a_prev_pad[start_verticle:end_verticle,start_horizontal:end_horizontal,:]
                    
                    Z[i, h, w, c] = conv_single_forward(a_slice_prev,W[:,:,:,c], b[:,:,:,c])
    
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = A_prev
    
    return Z, cache




def conv_multi_backward(dZ,cache,W,b,stride,pad):
   
    # Retrieve information from "cache"
    
    A_prev=cache
    # Retrieve dimensions from A_prev's shape
    (m, h_prev, w_prev, n_c_prev) = np.shape(A_prev)
    
    # Retrieve dimensions from W's shape
    (f, f, n_c_prev, n_C) = np.shape(W)
    
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = np.shape(dZ)
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, h_prev, w_prev, n_c_prev))                            
    dW = np.zeros((f, f, n_c_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                      
        
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           
                    
                    start_verticle = h * stride
                    end_verticle = h * stride+ f
                    start_horizontal = w * stride
                    end_horizontal = w * stride + f
                    
                    a_slice = a_prev_pad[start_verticle:end_verticle,start_horizontal:end_horizontal,:]

                    da_prev_pad[start_verticle:end_verticle, start_horizontal:end_horizontal, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
            
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        #dA_prev=relu_der(dA_prev,cache)
        #db = db -0.1*db
        #dW = dW -0.1*dW
    assert(dA_prev.shape == (m, h_prev, w_prev, n_c_prev))
    
    return dA_prev, dW, db





def classify(X, W,b,stride,pad):
   
    caches = []
    A = X

    AL,caches=conv_multi_forward(A,W,b,stride,pad)
       
    Ypred=AL
    return Ypred

def cnn(X,Y,W,b,stride,pad,num_iterations):
    
    
    A0 = X
    costs = []
    train_error=[]
    for ii in range(num_iterations):
        
        AL, cache = conv_multi_forward(X,W,b,1,1)
        AL=relu(AL)
        AL=softmax_fc(AL)
        AL=relu_der(AL,cache)
        AL,W,b = conv_multi_backward(AL,cache,W,b,1,1)
        pred=classify(X,W,b,1,1)
        train_error.append(softmax_loss_backwards(pred,X))
            #print("Cost at iteration %i is: %.05f" %(ii, cost))
    parameters=W,b
    #return costs, parameters,train_error
    return parameters,train_error

def main():
    

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = \
            mnist(ntrain=100,ntest=10,digit_range=[0,10])
    # initialize learning rate and num_iterations

    
    num_iterations =  4
    
    
    train_data = train_data.reshape(train_data.shape[1], 28,28,1)
    train_data = train_data.astype(np.float32)
    train_data = np.multiply(train_data, 1.0 / 255.0)

    test_data = test_data.reshape(test_data.shape[1], 28,28,1)
    test_data = test_data.astype(np.float32)
    test_data = np.multiply(test_data, 1.0 / 255.0)
    
    
    
    # compute the accuracy for training set and testing set
    W,b=initialize_parameters(filter_size=3,filters=5)
    parameters,train_error = cnn(train_data, train_label,W,b,1,1,num_iterations)
    W=parameters[0]
    b=parameters[1]
    train_Pred = classify(train_data,W,b,1,1)
    test_Pred = classify(test_data,W,b,1,1)
    
    trAcc = np.abs(100-(np.mean(np.abs(train_Pred - train_data)))/np.sum(train_data))
    teAcc = np.abs(100-(np.mean(np.abs(test_Pred - test_data)))/np.sum(test_data))
    print("Accuracy for training set is {0:0.3f}".format(trAcc))
    print("Accuracy for testing set is {0:0.3f}".format(teAcc))

    
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.title("error vs iteration")
    plt.plot(range(0,num_iterations),train_error)
    plt.show()

if __name__ == "__main__":
    main()
from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_examples = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    for i in range(num_examples):
        scores = X[i].dot(W)
        scores -= np.max(scores) # numeric stability
        correct_score = np.exp(scores[y[i]])
        sum_scores = 0.0
        for score in scores:
            sum_scores += np.exp(score)
            
        for j in range(len(scores)):
            p = np.exp(scores[j])/sum_scores
            dW[:, j] += ((p-(j == y[i])) * X[i, :])
       
        loss += -1*np.log(correct_score/sum_scores)
        
    loss /= num_examples
    loss += reg * np.sum(W * W)
    
    dW /= num_examples
    dW += reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_examples = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    scores -= np.max(scores,axis=1)[:, np.newaxis]
    correct_scores = np.exp(scores[np.arange(num_examples),y])
    sum_scores = np.sum(np.exp(scores),axis=1)
    
    # gradient calc
    p = np.divide(np.exp(scores),sum_scores[:, np.newaxis])
    p[np.arange(num_examples),y] -= 1
    dW = X.T.dot(p)
    dW /= num_examples
    dW += reg*W
    
    
    
    
    # loss calc
    loss = np.sum(-1*np.log(np.divide(correct_scores,sum_scores)))
    loss /= num_examples
    loss += reg * np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

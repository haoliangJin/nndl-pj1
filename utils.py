import numpy as np


def sigmoid(input):
    return 1/(1+np.exp(-input))


def d_sigmoid(input):
    return input*(1-input)


def relu(input):
    return np.maximum(input,0)


def d_relu(input):
    # input: batch*size
    #print(input.shape)
    n1,n2=input.shape[0],input.shape[1]
    ans=np.zeros((n1,n2))
    ans[input>0]=1
    return ans


def cross_entropy(x,y):
    # x: batch*num_class
    # y: batch
    n=x.shape[0]
    shift_x=x-np.max(x,axis=1,keepdims=True)
    z=np.sum(np.exp(shift_x),axis=1,keepdims=True)
    log_prob=shift_x-np.log(z)
    probs=np.exp(log_prob)
    loss=-np.sum(log_prob[np.arange(n),y])
    dx=probs.copy()
    dx[np.arange(n),y]-=1
    return loss, dx


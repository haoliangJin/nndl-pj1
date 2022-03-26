import argparse
import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import decode_idx1_ubyte,decode_idx3_ubyte
from utils import relu,d_relu,cross_entropy
from sklearn.model_selection import train_test_split as split
import pickle
import time


class net():
    def __init__(self,lr,input_size,hidden_size,out_size,decay_rate,epochs,act,d_act,loss_func,scale=1,lr_deacy=0.97):
        self.lr=lr
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.out_size=out_size
        self.decay_rate=decay_rate
        self.epochs=epochs
        self.relu=act
        self.d_relu=d_act
        self.loss_func=loss_func
        self.scale=scale
        self.lr_decay=lr_deacy

        self.params={}
        self.init_flag=False
        self.loss=[]
        self.test_loss=[]
        self.test_acc=[]

        if not self.init_flag:
            self.init_weight()

    def init_weight(self):
        self.params['w1']=np.random.randn(self.input_size,self.hidden_size)*self.scale
        self.params['w2']=np.random.randn(self.hidden_size,self.out_size)*self.scale
        self.params['b1']=np.zeros(self.hidden_size)
        self.params['b2']=np.zeros(self.out_size)
        self.init_flag=True

    def train(self,input,y,test_x,test_y):
        n = input.shape[0]
        start=time.time()
        for epoch in range(self.epochs):
            # forward
            h1=np.dot(input,self.params['w1'])+self.params['b1']
            h1=self.relu(h1)
            h2=np.dot(h1,self.params['w2'])+self.params['b2']
            output=self.relu(h2)

            # loss
            loss,dy=self.loss_func(output,y)
            loss/=n
            dy/=n
            regularization = np.sum(np.square(self.params['w1'])) + np.sum(np.square(self.params['w2']))
            loss+=regularization*self.decay_rate*0.5
            self.loss.append(loss)
            print('epoch %d, train_loss %.4f' % (epoch+1,loss))

            # batch gradient
            dout=dy*self.d_relu(output)

            dw2=np.dot(h1.T,dout) + self.decay_rate * self.params['w2']  # l2_regularization
            db2=np.sum(dout,axis=0)
            dh1=np.dot(dout,self.params['w2'].T)*self.d_relu(h1)

            dw1=np.dot(input.T,dh1) + self.decay_rate * self.params['w1']  # l2_regularization
            db1=np.sum(dh1,axis=0)

            self.params['w2'] -= dw2 * self.lr
            self.params['b2'] -= db2 * self.lr
            self.params['w1'] -= dw1 * self.lr
            self.params['b1'] -= db1 * self.lr

            # lr decay
            self.lr*=self.lr_decay

            # test_acc & test_loss
            test_loss,test_acc=self.test(test_x,test_y)

            print('test_loss %.4f,test_acc %.4f' % (test_loss, test_acc))
            self.test_acc.append(test_acc)
            self.test_loss.append(test_loss)

        print('time_cost %.4f' % (time.time()-start))
        return self.loss,self.test_loss,self.test_acc


    def test(self,input,y):
        # forward
        h1 = np.dot(input, self.params['w1']) + self.params['b1']
        h1 = self.relu(h1)
        h2 = np.dot(h1, self.params['w2']) + self.params['b2']
        output = self.relu(h2)
        #output = np.dot(h2, self.params['w3']) + self.params['b3']
        #output = self.relu(output)

        # loss
        n=output.shape[0]
        loss, dy = self.loss_func(output, y)
        loss/=n

        # acc
        ans=0
        y_pred=np.argmax(output,axis=1)
        n=y_pred.shape[0]
        for i in range(n):
            if y_pred[i]==y[i]:
                ans+=1
        ans/=n
        return loss,ans

    def get_params(self):
        return self.params

    def get_loss(self):
        return self.loss,self.test_loss,self.test_acc


if __name__ == '__main__':
    # hyper-parameter setting
    parser=argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=0.35)
    parser.add_argument('--decay_rate',type=float,default=4e-5)
    parser.add_argument('--lr_decay',type=float,default=0.99)
    parser.add_argument('--hidden_size',type=int,default=256)
    parser.add_argument('--output_class',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--scale',type=float,default=0.1)
    parser.add_argument('--seed',type=int,default=2022)

    args=parser.parse_args()

    # data loading
    train_images_idx3_ubyte_file = 'data/train-images.idx3-ubyte'
    train_labels_idx1_ubyte_file = 'data/train-labels.idx1-ubyte'
    test_images_idx3_ubyte_file = 'data/t10k-images.idx3-ubyte'
    test_labels_idx1_ubyte_file = 'data/t10k-labels.idx1-ubyte'

    train_images=decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels=decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    train_labels=np.array(train_labels,dtype=np.int)
    test_images=decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels=decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    test_labels=np.array(test_labels,dtype=np.int)

    # normalization
    n1,n2=train_images.shape[0],test_images.shape[0]
    train_images=train_images.reshape((n1,-1))/255
    test_images=test_images.reshape((n2,-1))/255

    # training
    np.random.seed(args.seed)  # random seed control
    train_x,valid_x,train_y,valid_y=split(train_images,train_labels,test_size=0.3,random_state=args.seed)  # train_set and valid_set
    input_size=train_images.shape[1]
    # parameters searching
    best_model=None
    best_loss=float('inf')
    for args.lr in [0.35,0.3,0.25]:
        for args.hidden_size in [256,128]:
            for args.decay_rate in [4e-5,1e-5]:
                model=net(args.lr,input_size,args.hidden_size,args.output_class,args.decay_rate,args.epochs,relu,d_relu,cross_entropy,args.scale,args.lr_decay)
                _,valid_loss,_=model.train(train_x,train_y,valid_x,valid_y)
                if valid_loss[-1]<best_loss:
                    print('best found')
                    best_loss=valid_loss[-1]
                    best_model=model

    # result plotting
    loss,valid_loss,valid_acc=best_model.get_loss()
    x=np.arange(1,args.epochs+1)
    plt.plot(x,loss,label='train_loss')
    plt.plot(x,valid_loss,label='valid_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss.png')

    plt.close('all')
    plt.plot(x,valid_acc,label='valid_acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.savefig('test_acc.png')

    # model saving
    with open('model.pkl','wb') as f:
        pickle.dump(best_model,f)



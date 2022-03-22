import numpy as np
import pickle
from utils import relu
from mnist_loader import decode_idx1_ubyte,decode_idx3_ubyte
from main import net
#
# # model loading
# with open('params.pkl','rb') as f:
#     params=pickle.load(f)
#
#
# def test(input,y):
#     # forward
#     h1 = np.dot(input, params['w1']) + params['b1']
#     h1 = relu(h1)
#     h2 = np.dot(h1, params['w2']) + params['b2']
#     h2 = relu(h2)
#     output = np.dot(h2, params['w3']) + params['b3']
#     output = relu(output)
#
#     # calculate test acc
#     ans=0
#     y_pred=np.argmax(output,axis=1)
#     n=y_pred.shape[0]
#     for i in range(n):
#         if y_pred[i]==y[i]:
#             ans+=1
#     ans/=n
#     print('test_acc %.4f' % ans)
#     return ans

# test_data loading
test_images_idx3_ubyte_file = 'data/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'data/t10k-labels.idx1-ubyte'
test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
test_labels = np.array(test_labels, dtype=np.int)
n2=test_images.shape[0]
test_images = test_images.reshape((n2, -1)) / 255

# test(test_images,test_labels)

with open('model.pkl','rb') as f:
    model=pickle.load(f)
test_loss,test_acc=model.test(test_images,test_labels)
print('test_loss %.4f,test_acc %.4f' % (test_loss, test_acc))
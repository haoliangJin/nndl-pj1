import numpy as np
import pickle
from utils import relu
from mnist_loader import decode_idx1_ubyte,decode_idx3_ubyte
from main import net
from matplotlib import pyplot as plt

# test_data loading
test_images_idx3_ubyte_file = 'data/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'data/t10k-labels.idx1-ubyte'
test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
test_labels = np.array(test_labels, dtype=np.int)
n2=test_images.shape[0]
test_images = test_images.reshape((n2, -1)) / 255

# testing result
with open('model.pkl','rb') as f:
    model=pickle.load(f)
test_loss,test_acc=model.test(test_images,test_labels)
print('test_loss %.4f,test_acc %.4f' % (test_loss, test_acc))

# visualize parameter
params=model.get_params()

plt.matshow(params['w1'][:51,:51])
plt.title('first matrix')
plt.legend()
plt.savefig('first.png')
plt.matshow(params['w2'][:21,:11])
plt.title('second matrix')
plt.savefig('second.png')



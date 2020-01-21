# In[1]:
import numpy as np
import pandas as pd

# In[2]:

from tensorflow.keras.models import load_model

# In[3]:
img_width, img_height = 256, 256

# In[4]:
test_preprocessed_images = np.load('../test_preproc_CNN.npy')

# In[5]:
#Define Path
model_path = '../CNN_best_weights_256.h5'
#model_path = '../CNN_augmentation_best_weights_256.h5'
#model_path = '../vgg16_best_weights_256.h5'
#model_path = '../vgg16_drop_batch_best_weights_256.h5'
#model_path = '../vgg19_drop_batch_best_weights_256.h5'
#model_path = '../resnet101_drop_batch_best_weights_256.h5'

#Load the pre-trained models
model = load_model(model_path)


# In[6]:
#Prediction Function
array = model.predict(test_preprocessed_images, batch_size=1, verbose=1)
answer = np.argmax(array, axis=1)

# In[7]:
test_df = pd.read_csv('../dataset/test.csv')
y_true = test_df['labels']
y_pred = array

# In[8]:
from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)



import numpy as np
import glob
import matplotlib.image as img
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV as GridSearch

class_names = ['Background','Material','Pores','Inclusion']
feature_names = ['Grayscale Intensity','Mean']
num_classes = len(class_names); num_features = len(feature_names)


def local_meanr(test):
    Shape = test.shape
    new = np.zeros((Shape[0], Shape[1]))
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            if i == 0 and j == 0:  # left-top-corner
                new[i, j] = (test[i, j] + test[i + 1, j] + test[i + 1, j + 1] + test[i, j + 1]) / 4
            elif i == Shape[0] - 1 and j == 0:  # left-bottom-corner
                new[i, j] = (test[i, j] + test[i - 1, j] + test[i - 1, j + 1] + test[i, j + 1]) / 4
            elif i == Shape[0] - 1 and j == Shape[1] - 1:  # rigth-bottom-corner
                new[i, j] = (test[i, j] + test[i - 1, j] + test[i - 1, j - 1] + test[i, j - 1]) / 4
            elif i == 0 and j == Shape[1] - 1:  # right-top-corner
                new[i, j] = (test[i, j] + test[i + 1, j] + test[i + 1, j - 1] + test[i, j - 1]) / 4




            elif j == 0 and i > 0 and i < Shape[1] - 1:  # left-edge
                new[i, j] = (test[i, j] + test[i, j + 1] + test[i - 1, j + 1] + test[i + 1, j] + test[i + 1, j + 1] +
                             test[i - 1, j]) / 6
            elif j == Shape[1] - 1 and i > 0 and i < Shape[0] - 1:  # rigth-edge
                new[i, j] = (test[i, j] + test[i, j - 1] + test[i + 1, j] + test[i - 1, j] + test[i - 1, j - 1] + test[
                    i + 1, j - 1]) / 6
            elif i == Shape[0] - 1 and j > 0 and j < Shape[1] - 1:  # bottom-edge
                new[i, j] = (test[i, j] + test[i - 1, j] + test[i, j - 1] + test[i, j + 1] + test[i - 1, j + 1] + test[
                    i - 1, j - 1]) / 6
            elif i == 0 and j > 0 and j < Shape[1] - 1:  # top-edge
                new[i, j] = (test[i, j] + test[i + 1, j] + test[i + 1, j + 1] + test[i + 1, j - 1] + test[i, j - 1] +
                             test[i, j + 1]) / 6




            elif j > 0 and j < Shape[1] - 1 and i > 0 and i < Shape[0] - 1:  # middle
                new[i, j] = (test[i, j] + test[i, j + 1] + test[i, j - 1] + test[i - 1, j] + test[i + 1, j] + test[
                    i + 1, j + 1] + test[i - 1, j - 1] + test[i - 1, j + 1] + test[i + 1, j - 1]) / 9

    return new

def flatten(Mat):
    S = Mat.shape
    mat = Mat.reshape(S[0]*S[1],1)
    return mat




image_list = []
for filename in glob.glob('/Users/harshbordekar/Desktop/untitled folder 2/probe3/*.jpg'):
    im = img.imread(filename)
    image_list.append(im)

len(image_list)


def datamine(gud):
    gud1 = flatten(gud)
    gud2 = flatten(local_meanr(image_list[0]))
    sub = np.append(gud1, gud2, axis=1)

    return sub

s = image_list[0].shape
maha = np.zeros((s[0]*s[1],2,len(image_list)))

for i in range(6):#range(len(image_list)):
    bharat = datamine(image_list[i])
    maha[:,:,i] = bharat
    del bharat


def get_labels_as_image(Y, shape):
    # Convert labels Y back to an color image
    Y = Y.reshape((shape[0], shape[1], 1))
    Y = edge_effect(Y)
    new_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if Y[i, j] == 0:
                new_image[i, j] = [0, 0, 0]  # black
            elif Y[i, j] == 1:
                new_image[i, j] = [255, 0, 0]  # red

            elif Y[i, j] == 2:
                new_image[i, j] = [0, 255, 0]  # green
            elif Y[i, j] == 3:
                new_image[i, j] = [0, 0, 255]  # blue
            elif Y[i, j] == 4:
                new_image[i, j] = [255, 255, 255]  # blue
    return new_image


new_header = ['value','Mean','target']
df = pd.read_csv("train16_4.csv",header=None,names=new_header)
df0 = df['value'].values.reshape(-1,1)
array1 = np.array(df0)
df1 = df['Mean'].values.reshape(-1,1)
array2 = np.array(df1)
X = np.append(array1,array2,axis=1)
df4 = df['target'].values
y = np.array(df4)
print(X.shape)

################ model############
# The fixed parameters are hold constant
fixed_parameters = {
    'max_features' : 'sqrt',   # Number of features considered per node: 'square rule'
    'criterion' : 'entropy'    # Splitting criterion: 'information gain'
}

# The tuned parameters are optimized during the grid search.
# Instead of a fixed value, we store a range of values for each variable
tunable_parameters = {
    'max_depth': range(2,4),        # Maximum depth of the tree
    'min_samples_split': range(2,4)  # Min. number of samples in a node to continue splitting
}

# Create an instance of the model that is to be optimized
model = DecisionTreeClassifier(**fixed_parameters)

# Create the optimizer and run the optimization
opt = GridSearch(model, tunable_parameters, cv = 3, scoring="accuracy", verbose=1)
opt.fit(X_train,y_train)


# Save and print optimal parameters
opt_parameters = opt.best_params_
print("Best found parameters:", opt_parameters)
############################################################################################


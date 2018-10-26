---
title: Insert title here
key: 4c01cfa1ac1c3806f2f783e0b2d7ffd2

---
## Your First Neural Networks

```yaml
type: "TitleSlide"
key: "681fa29a0d"
```

`@lower_third`

name: Mahmoud ElHussieni
title: Instructor at Datacamp


`@script`



---
## Classifying the Iris Data Set with Keras

```yaml
type: "FullSlide"
key: "f15252eb97"
```

`@part1`
_**Firstly**,_ let's load our packages:
```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
```
**_Secondly_**, let's load our dataset and assign the features to X and the class label to y
```
iris_data = load_iris()
x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column
```


`@script`



---
## Exploring Iris

```yaml
type: "FullSlide"
key: "a6c35f0553"
```

`@part1`
Let's take a look to Iris data set and its features
![](https://raw.githubusercontent.com/MElHussieni/ShinyApp-with-Iris/master/Screenshot%20from%202018-10-24%2010-42-46.png)


`@script`



---
## Building Our Keras model

```yaml
type: "FullSlide"
key: "24bf4864ee"
```

`@part1`
**_Thirdly_,** Before feeding our data to Neural net we should convert categorical features to numeric
Here we can use [One-Hot Encoding technique](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f), and then split the data to Training data and Testing data
```
# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
```
**_Fourthly_**,Now Let's build our model
```
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

```


`@script`



---
## Train and Evaluate Deep Learning with Keras

```yaml
type: "FullSlide"
key: "8b597ab8f9"
```

`@part1`
Before Training, the Sequential model has a summary function to see the summary of the network we build  
```
Neural Network Model Summary: 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
fc1 (Dense)                  (None, 10)                50        
_________________________________________________________________
fc2 (Dense)                  (None, 10)                110       
_________________________________________________________________
output (Dense)               (None, 3)                 33        
=================================================================
Total params: 193
Trainable params: 193
Non-trainable params: 0
```
```
# Train the model
model.fit(train_x, train_y,epochs=200)
# Test on unseen data
results = model.evaluate(test_x, test_y)
```


`@script`



---
## The Training Results

```yaml
type: "FullSlide"
key: "9b67db5787"
```

`@part1`
After training 200 epoch, Our model achieves 96% Test accuracy and 0.03 loss
```
Epoch 1/200
 - 1s - loss: 1.2244 - acc: 0.3333
Epoch 2/200
 - 0s - loss: 1.0287 - acc: 0.5667
.
.
.
Epoch 198/200
 - 0s - loss: 0.0730 - acc: 0.9750
Epoch 199/200
 - 0s - loss: 0.0722 - acc: 0.9750
Epoch 200/200
 - 0s - loss: 0.0719 - acc: 0.9750
30/30 [==============================] - 0s 1ms/step
Final test set loss: 0.034482
Final test set accuracy: 0.966667
```
_Conguratlisations we build our First Deep Learning model_


`@script`



---
## Let's Practice!

```yaml
type: "FinalSlide"
key: "b210c8df50"
```

`@script`



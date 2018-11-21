# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/lenet_model_failure.png "LeNet Model Failure"
[image2]: ./images/nvidia_model_failure_1.png "Nvidia Model Failure 1"
[image3]: ./images/nvidia_model_failure_2.png "Nvidia Model Failure 2"
[image4]: ./images/loss_metrics_nvidia_model_batch_32_epoch_7_1542728717.png "Nvidia model loss metrics"
[image5]: ./images/cnn-architecture-624x890.png "Nvidia Model Architecture"
[image6]: ./images/2018_11_20_16_37_19_117.jpg "Camera Image"
[image7]: ./images/lenet.png "Lenet Architecture"
[image8]: ./images/model.png "Nvidia Model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

My Project Github: https://github.com/ysonggit/CarND-Behavioral-Cloning-P3

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

##### Usage:

```
$ python model.py -h
usage: model.py [-h] [-m MODEL] [-b BATCHSIZE] [-e EPOCHS]

Model Trainer

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model name
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batch size
  -e EPOCHS, --epochs EPOCHS
```

For the argument `-m`, the parameter is either `lenet` or `nvidia`. The model.py can train the LeNet model or the Nvidia model introduced in the lectures and the project tutorial.

The default batch size is 32, the same as the value used by the Keras [1]. The default epoch number is 3.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The final successful autonomous drive is conducted by the Nvidia model [2]. My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (function `createNvidiaModel()` in model.py, lines 32-46).

The Nvidia CNN architecture is shown in the figure below [2]:
![Nvidia model][image5]

The model includes RELU layers to introduce nonlinearity (code lines 36-43), and the data is normalized in the model using a Keras lambda layer (code line 34).

Meanwhile, the model has a cropping layer to crop the original camera image (figure below shows an example camera image). The crop parameter (50, 20), (0, 0) are very important values to make the trained model work well.

![Camera Image][image6]

#### 2. Attempts to reduce overfitting in the model

In the LeNet model, I applied the dropout and the max pooling layers to reduce the overfitting (function `createLeNetModel()`, code line 16). However, I test the model with the simulator but the car still drives away from the road.

My Nvidia model contains max pooling layers after each RELU layer, but it does not contain any dropout regularizations. Because Darien Martinez proved that his solution can train the Nvidia model very rapidly in only 3 epochs without using any regularizations in the neural network [4].

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 135).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the project tutorial videos, and my solution is mostly based on the idea and codes shown in the videos.

My first step was to use a LeNet model (figure below) [5] similar to the one I learned from the LeNet Lab. The difference is that, here, with Keras, it is much easier to construct the entire model using only 15 lines of codes (code lines 16-30).

![lenet architecture][image7]

I thought this model might not be the best solution, otherwise, there is no need to introduce a "more powerful model". After working on this model by tuning the parameters, such as the epoch number, batch size, pre-processing the captured camera image, etc., the simulated autonomous drive always failed by deviating from the road, like what the below snapshot is displaying.

![Lenet simulation][image1]

The "Plan B" is to train the Nvidia model [2]. As the most of the codes of the function `createNvidiaModel()` are given in the tutorial, my work focused on testing and tuning the training process according to the simulations performances.

The trained Nvidia model works well most of the time by leading the car's movement along the center of the lane. However, there are two places the car lost itself as the pictures below are showing:

![failure 1][image2]
![failure 2][image3]

This actually turns to be the most difficult part of this project. The key to conquer this problem is doing the data augmentation. I would like to describe my steps the the section 3.

#### 2. Final Model Architecture

The final Nvidia model architecture (model.py lines 32-46) consisted of a convolution neural network with the following layers and layer sizes:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![model graph][image8]

#### 3. Creation of the Training Set & Training Process

I observe that the common fact between two accidental scenarios, where the car drives out of the road, is that at those spots, the lane lines are not simple yellow lines but with red-white strips. Inspired by the work of Alex Staravoitau [3] and Darien Martinez [4], I augmented the image data using two methods to improve the driving behavior in these cases:
1. Horizontal flip the images (code lines 82-86)
2. Combine center camera image with left and right camera images (code lines 124-128)

With the data augmentation process, the trainer has more data points to learn from, that is, 48216 of 160x320x3 images. The simulations prove that the data augmentation is the key to the success of training the Nvidia model.

Finally I randomly shuffled the data set and put 20% of the data into a validation set: 38572 out of 48216 images are used for training, and the rest 9644 samples are used for validating. I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

```
15:44 $ python model.py -m nvidia -e 7
Using TensorFlow backend.
Build Nvidia Model ...
Read data from drive logs ...
('X_train.shape = ', (48216, ))
('y_train.shape = ', (48216,))
Start training nvidia model: batch size = 32, epochs = 7
Train on 38572 samples, validate on 9644 samples
Epoch 1/7
2018-11-20 15:46:25.484024: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
38572/38572 [==============================] - 612s 16ms/step - loss: 0.0179 - acc: 0.1802 - val_loss: 0.0184 - val_acc: 0.1837
Epoch 2/7
38572/38572 [==============================] - 549s 14ms/step - loss: 0.0137 - acc: 0.1802 - val_loss: 0.0191 - val_acc: 0.1837
Epoch 3/7
38572/38572 [==============================] - 592s 15ms/step - loss: 0.0117 - acc: 0.1802 - val_loss: 0.0200 - val_acc: 0.1837
Epoch 4/7
38572/38572 [==============================] - 739s 19ms/step - loss: 0.0105 - acc: 0.1802 - val_loss: 0.0212 - val_acc: 0.1837
Epoch 5/7
38572/38572 [==============================] - 663s 17ms/step - loss: 0.0095 - acc: 0.1802 - val_loss: 0.0212 - val_acc: 0.1837
Epoch 6/7
38572/38572 [==============================] - 562s 15ms/step - loss: 0.0087 - acc: 0.1802 - val_loss: 0.0203 - val_acc: 0.1837
Epoch 7/7
38572/38572 [==============================] - 610s 16ms/step - loss: 0.0080 - acc: 0.1802 - val_loss: 0.0201 - val_acc: 0.1837
Training took 4326.44630909 seconds.
Model saved as nvidia_model_batch_32_epoch_7_1542728717.h5
```

I plot the train set loss and the validation set loss below. The training set loss looks good because it decreases as the epochs increase. But the validation set loss does not change much. Unfortunately, I am not able to find the root cause why the validation set loss does not decrease as epochs increase.
![loss plot][image4]

I implemented and verified the algorithm of Darien Martinez [4] and feel very surprised that the model does a great job of driving the car after only 3 epochs of training.

Moreover, I also want to know if the trained Nvidia model would perform better when tuning the parameters, such as increasing batch size from 32 to 64, or increase the epochs from 3 to 7, or 10. By comparing the result simulations, there are no significant difference among those results. In other words, tuning the number of epochs or batch size would not improve as much performance as adopting proper data augmentation methods.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Here is the link to the output simulation [video](./video.mp4) of the autonomous drive.


## References
[1] [Keras Documentation](https://keras.io/models/model/)
[2] [Nvidia Blog: End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
[3] [Alex Staravoitau, End-to-End Learning for Self-driving Cars](https://navoshta.com/end-to-end-deep-learning/)
[4] [Darien Martinez's Github  ](https://github.com/darienmt/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md)
[5] [LeCun et al., LeNet-5 Convolutional Neural Networks](http://yann.lecun.com/exdb/lenet/)

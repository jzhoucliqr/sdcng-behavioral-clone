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

[image1]: ./examples/recover_left.jpg "Recover from left"
[image2]: ./examples/recover_right.jpg "Recover from right"
[image3]: ./examples/bridge.jpg "Into the bridge"
[image4]: ./examples/corner.jpg "Driving around the corner"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 is the video created by auto mode on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I'm using the [Nvidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 5 convolutional layers and 3 fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in the first 3 convolutional layers, in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used mse for loss function and adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving from both clock wise and anti clock wise, and also recovering from the left and right sides of the road. I also collect more data for driving arond the corners and into the bridge. Because initially although the model can drive the car most of the time, I see it hits the bridge couple of times and stuck there. 

![alt text][image3]

![alt text][image4]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I first followed the lecture to have a functional code with a dead simple model, then I tried the lanet model, and then go with the nvidia model. 

Honestly there is not much design strategy for the model itself, I just go with it. I spent most of my time collecting data and training / testing. 

#### 2. Final Model Architecture

The final model architecture is the one from Nvidia paper, it consistes of a convolution neural network with the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image  | 
| Cropping2D         		| cropping((70,25),(0,0))  | 
| Convolution 5x5    	| (24, 5, 5), subsample (2,2), with 'relu' activation  |
| Convolution 5x5    	| (36, 5, 5), subsample (2,2), with 'relu' activation  |
| Convolution 5x5    	| (48, 5, 5), subsample (2,2), with 'relu' activation  |
| Convolution 3x3    	| (64, 3, 3),  with 'relu' activation  |
| Convolution 3x3    	| (64, 3, 3),  with 'relu' activation  |
| Flatten    	|   |
| Dense    	|  100 |
| Dense    	|  50 |
| Dense    	|  10 |
| Dense    	|  1 |



#### 3. Creation of the Training Set & Training Process

Even driving manually is not easy to keep the car in lane the whole time. I tried with keyboard many times, but couldn't get enough data without going out of lane. 

Then I use a mouse, set the moving speed to the lowest, then finally feel better at driving manually. So I collected several laps of both clock-wise and anti-clock wise, then some data for both recovering from left and right side of lane, like the following image shows. I also tried drive towards the side then back to the center couple of times.

I tried to collect data on the second lane, but couldn't do with good results. So in the end i only use data from the first lane.   

![alt text][image1]

![alt text][image2]

Finally, together with the sample data, I have 22163 samples. Then I use the three cameras images, and also flip the image, so in total I have 22163*6 = 132978 images. Then I split with 0.2 validation set, so total training sample is 106380, validation sample is 26598. 

I use an AWS GPU g2.2xlarge instance to train. Without generator, even this 16GB memory was not enough, I tried with a 64 GB instance which works fine. Then I implemented with generator, and it only takes about 2 GB memory. 

I tried with different epochs, the training almost converge with just 2 or 3 epoch. So in the end I'm using 3 epoch. Each epoch takes about 3 minutes to finish training, so together with validation the total trainig time was about 13 minutes.

```
Epoch 1/3
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GRID K520
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:00:03.0
Total memory: 3.94GiB
Free memory: 3.91GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: 0000:00:03.0)
106380/106380 [==============================] - 187s - loss: 0.0236 - val_loss: 0.0214
Epoch 2/3
106380/106380 [==============================] - 184s - loss: 0.0185 - val_loss: 0.0164
Epoch 3/3
106380/106380 [==============================] - 186s - loss: 0.0160 - val_loss: 0.0156

real	9m21.572s
user	9m23.020s
sys	2m50.268s
```

#### 4. Simulation

I've done lots of tests, the final result although is not perfect, but was able to stay on track for all the test I've run. 

The video.mp4 contains about 3 laps of simulation, sometimes it looks like the car is going out of track, but every time it was able to manage stay on track. 

It runs on the blue lines couple of times, but actually I don't blame the model, because in the training data I was not able to drive without runing on the blue line several times. 



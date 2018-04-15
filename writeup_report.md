# **Behavioral Cloning** 

**Behavioral Cloning Project**

## Videos

My videos are available at 

[track1 - lake](https://www.youtube.com/watch?v=5q5TRMQNtVM)
[track2 - jungle](https://youtu.be/6vMxVQRcevc)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[track1_distribution]: ./report/track1_distribution.png "Track1 Data Distribution Histogram"
[track2_distribution]: ./report/track2_distribution.png "Track2 Data Distribution Histogram"
[track1_normal]: ./report/track1_normal "Track1 normal"
[track1_translated]: ./report/track1_translated "Track1 translated"
[track1_flipped]: ./report/track1_flipped.jpg "Track1 flipped"
[track2_normal]: ./report/track2_normal.jpg "Track2 normal"
[track2_translated]: ./report/track2_translated.jpg "Track2 translated image"
[track2_flipped]: ./report/track2_flipped.jpg "Track2 flipped image"
[track1_recovery_1]: ./report/track1_recovery_1.jpg "Track1 recovery 1"
[track1_recovery_2]: ./report/track1_recovery_2.jpg "Track1 recovery 2"
[track1_recovery_3]: ./report/track1_recovery_3.jpg "Track1 recovery 3"
[track1_recovery_4]: ./report/track1_recovery_4.jpg "Track1 recovery 4"
[track1_recovery_5]: ./report/track1_recovery_5.jpg "Track1 recovery 5"
[track1_recovery_6]: ./report/track1_recovery_6.jpg "Track1 recovery 6"
[track1_recovery_7]: ./report/track1_recovery_7.jpg "Track1 recovery 7"
[track1_recovery_8]: ./report/track1_recovery_8.jpg "Track1 recovery 8"
[track2_recovery_1]: ./report/track2_recovery_1.jpg "Track1 recovery 1"
[track2_recovery_2]: ./report/track2_recovery_2.jpg "Track1 recovery 2"
[track2_recovery_3]: ./report/track2_recovery_3.jpg "Track1 recovery 3"
[track2_recovery_4]: ./report/track2_recovery_4.jpg "Track1 recovery 4"
[track2_recovery_5]: ./report/track2_recovery_5.jpg "Track1 recovery 5"
[track2_recovery_6]: ./report/track2_recovery_6.jpg "Track1 recovery 6"
[track2_recovery_7]: ./report/track2_recovery_7.jpg "Track1 recovery 7"
[track2_recovery_8]: ./report/track2_recovery_8.jpg "Track1 recovery 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network for track1 (lake track)
* model_hard.h5 containing a trained convolution neural network for track2 (jungle track) 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

OR

```sh
python drive.py model_hard.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the LeNet architecture that we have been using. Unlike many of my peers, i did not use the Nvidia architecture and it turns out a simpler model is able to achieve the level of complexity required to pass the assignment.

I followed Paul Heraty's tips and resized the input image by half, allowing huge training speedup without compromising functionality. I also added a lambda layer to normalize all values between -0.5, 0.5, and cropped off the top 35 pixels and bottom 13 pixels of image which is redundant. (code line 131 - 142)

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x160x3 RGB image   							| 
| Cropping         		| 32x160x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x156x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x78x6  				|
| Convolution 5x5	    | 1x1 stride, outputs 10x74x6 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x37x6 	 				|
| Dropout				| dropout 50%									|
| Fully connected		| input 1110, output  120     					|
| Fully connected		| input 120, output  84     					|
| Fully connected		| input 84, output  1     			 			|
|						|												|

The model includes RELU layers to introduce nonlinearity (code line 133, 136).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer right after the convolutional layers in order to reduce overfitting (model.py line 139). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 120-125). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with a custom learning rate of 0.0001. I decided to use a lower learning rate because the training and validation loss was making big jumps in different directions after every epoch. I was able to reach a lower validation loss of 0.83 versus 1.3 before making the change. As a result, of the lower learning rate, i increased the number of epochs by a factor of 10, to 50, and realized the validation loss converges around 25, hence my final number of epochs is 25.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a recorded 2 laps of center lane driving, and 1 lap of recovery driving from left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look at what others had done in the past. It seems that the consensus among udacity students is that the model architecture is not that important, what is really crucial is the quantity and quality of the training data.

My first step was to use the simplest convolutional neural network possible to accomplish the task and i decided to start with LeNet because it is well studied and known to work well for images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. A lower validation loss does not necessarily mean that it will perform better in the simulator. This is because it is really hard to get accurate training data. The human operator doing the steering cannot guarantee that he is giving the correct steering angle every single frame. He only needs to give correct steering angle on certain frames and the car will still stay on the road.

The final step was to run the simulator to see how well the car was driving around track one. The car was able to drive well on straight roads, but would get stuck at the bridge and sharp curves. After visualizing the distribution of steering angles in the dataset, i realized there is just not enough training data with high steering angles to deal with sharp turns. Hence, i recorded more training footage at the precise location where the car gets stuck and at sharp curves, evening out the data distribution. The final distribution still has steering angles near 0 over-represented, but it has enough high steering angles to get the job done.

Track 1 steering angle distribution:
![alt text][track1_distribution]

Track 2 steering angle distribution:
![alt text][track2_distribution]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 130-142) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x160x3 RGB image   							| 
| Cropping         		| 32x160x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x156x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x78x6  				|
| Convolution 5x5	    | 1x1 stride, outputs 10x74x6 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x37x6 	 				|
| Dropout				| dropout 50%									|
| Fully connected		| input 1110, output  120     					|
| Fully connected		| input 120, output  84     					|
| Fully connected		| input 84, output  1     			 			|
|						|												|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][track1_normal]
![alt text][track2_normal]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it drifts off the center lane. These images show what a recovery looks like starting from the left.

![alt text][track1_recovery_1]
![alt text][track1_recovery_2]
![alt text][track1_recovery_3]
![alt text][track1_recovery_4]
![alt text][track1_recovery_5]
![alt text][track1_recovery_6]
![alt text][track1_recovery_7]
![alt text][track1_recovery_8]


Then I repeated this process on track two in order to get more data points. Track 2 has more tricky sharp turns and I had to make sure i recorded data starting from a position such that the center lane is completely out of the camera field of view (FOV) and teach the car to use the side lane markers to guide itself back to the center lane. An example of a sequence for one such tricky turn is shown below:

![alt text][track2_recovery_1]
![alt text][track2_recovery_2]
![alt text][track2_recovery_3]
![alt text][track2_recovery_4]
![alt text][track2_recovery_5]
![alt text][track2_recovery_6]
![alt text][track2_recovery_7]
![alt text][track2_recovery_8]


To augment the data sat, I also flipped images and angles thinking that this would train the car to drive equally well in clockwise and counterclockwise directions. For example, here are 2 images that has been flipped:

![alt text][track1_flipped]
![alt text][track2_flipped]

I also used the left and right cameras to teach the car how to drive towards the center lane by adding a correction factor of 0.2 (found to work well empirically) to the steering angle.

I also borrowed [naokishibuya](https://github.com/naokishibuya/car-behavioral-cloning) suggestion to translate images horizontally by a random amt of pixels and multiply it by 0.002 and add it to the steering angle. This can help the model to follow curves.

After the collection process, I had 10,766 number of data points for track1 and 4237 for track2.

I then preprocessed this data by:
- top 35 (the sky and background) and bottom 13 pixels (car hood) is cropped off
- the images are resized to 80 x 160 x 3 (resized in half for faster training)
- the images are normalized (x/255 - 0.5)

Data augmentation
- choose center, left, right images
- left image steering angle +0.2, right image steering angle -0.2
- flip (laterally invert) all images and steering angle
- randomly translate image horizontally with steering angle adjusted by 0.002 per pixel shifted. Credits: @naokishibuya

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by the validation loss converging after 25. I used an adam optimizer.

## Videos

My videos are available at 

[track1 - lake](https://www.youtube.com/watch?v=5q5TRMQNtVM)
[track2 - jungle](https://youtu.be/6vMxVQRcevc)
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Model Visualization"
[image3]: ./examples/rec1.jpg "Recovery Image"
[image4]: ./examples/rec2.jpg "Recovery Image"
[image5]: ./examples/rec3.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"
[image8]: ./examples/center.jpg "Center Image"
[image9]: ./examples/left.jpg "Left Image"
[image10]: ./examples/right.jpg "Right Image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* report.md summarizing the results
* video.mp4 video of autonomus driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Approach and summary

The overall strategy for deriving a model architecture was to find the best structure to support such goal.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because of it

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 1. Architecture Description


My model Uses deep convolutional neural network that takes images captured by the 3-cameras in the simulator and returns the steering angles. I used the Nvidia architecture as shown below:

![alt text][image1]

 here's a detailed description of the model layers I used:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 BGR image           | 
| cropping          | Output 80x320x3               |
| Lambda            | Normalization                  |
| Convolution 5x5   | 2x2 stride, depth 24,regularization l2(0.01)  	|
| RELU				    	|	Activation											|
| Convolution 5x5   | 2x2 stride, depth 36, regularization l2(0.01) 	|
| RELU				    	|	Activation											|		
| Convolution 5x5   | 2x2 stride, depth 48, regularization l2(0.01) 	|
| RELU				    	|	Activation											|
| Convolution 3x3   |  depth 64, regularization l2(0.01) 	|
| RELU				    	|	Activation											|
| Convolution 3x3   |  depth 64, regularization l2(0.01) 	|
| RELU				    	|	Activation											|
| Flatten   	    	| Flatten  o/p of last conv layer	|		
| Dropout Layer 		| Probability - 0.5    |
| Fully connected		| Dense,  Output = 100    |
| Fully connected		| Dense,  Output = 50    |
| Dropout Layer 		| Probability - 0.5    |
| Fully connected		| Dense,  Output = 10    |
| Fully connected		| Dense,  Output = 1    |
|						|												|
|						|												|
 


I used 5- convolutional layers followed a flatten layer, then 4- fully connected layers and 2 dropout layers in between

My model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting after first layer of flatten and before fully connected layer with output 10. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model parameters are tuned as following. The Adam optimizer is used, the rate is set with default value.
Keras's Generator was used to feed batch data as an when required, so that all data does not have to be stored in the Memory. 

epochs = 15
batch_size = 32
loss = mse

#### 5. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. I used a combination of folowing data:

* two or three laps of center lane driving
* one lap of recovery driving from the sides
* one lap focusing on driving smoothly around curves


To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here are example image of : 

center camera

![alt text][image6]

left camera

![alt text][image7]

right camera

![alt text][image8]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive to the center from the sides. These images show what a recovery looks like starting from right :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data set, I also flipped images and angles (steering measurements) thinking that this would increase data and prevent underfitting. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by :
* Cropping the unecessary portions : eg. mountains and sky
* cropping the bottom hood of the car

the above just makes the network harder to train and adds confusion.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by validation and training loss decreasing till 15th e poch and then increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.

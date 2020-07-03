# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./new_images/visualization.png "Visualization"
[image2]: ./new_images/color.jpg "Color"
[image3]: ./new_images/grayscale.jpg "Grayscale"
[image4]: ./new_images/23.jpg "Traffic Sign 1"
[image5]: ./new_images/14.jpg "Traffic Sign 2"
[image6]: ./new_images/31.jpg "Traffic Sign 3"
[image7]: ./new_images/36.jpg "Traffic Sign 4"
[image8]: ./new_images/25.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of train and test data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because as stated in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) from the notebook, doing so would give better accuracy than using color images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

As a last step, I normalized the image data because doing so make it a lot easier and faster for optimizer to find the solution.

I did not generate additional data.

[//]: # (To add more data to the the data set, I used the following techniques because ... )

[//]: # (Here is an example of an original image and an augmented image:)

[//]: # (![alt text][image3])

[//]: # (The difference between the original data set and the augmented data set is the following ... )


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution_1 5x5    	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution_2 5x5    	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten	    	    | output = 14x14x6 + 5x5x16 = 1576				|
| Dropout				| Keep_prob = 0.8								|
| Fully connected_1		| output = 120     								|
| RELU					|												|
| Fully connected_2		| output = 84     								|
| RELU					|												|
| Dropout				| Keep_prob = 0.7								|
| Fully connected_3		| output = 43     								|
| Softmax				|           									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer, with batch_size = 128, epoches = 30, and learning_rate = 0.0008. I added dropout after flattening CONV1 and CONV2 outputs with keep_prob=0.8, and also after FC2 layer with keep_prob=0.7. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 93.2%
* test set accuracy of 92.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started with LeNet architecture from class, because it is suggested in the notebook.

* What were some problems with the initial architecture?
When experimenting with LeNet architecture, the best validation accuracy I reached by tuning learning_rate is around 60~70%, which does not meet the project requirement, which is at least 93%. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I did not choose a different model architecture, but made following modifications based on LeNet. 
First, I looked into the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) in the notebook, and fed both the CONV1 outputs and CONV2 outputs into flatten layer, so that the low level features (in CONV1) of data can be taken account into in later layers.
Next, I added dropout after flatten layer, and after fully_connected_2 layer. I set keep_prob = 0.8 and 0.7 for them respectively after experimenting.

* Which parameters were tuned? How were they adjusted and why?
- learning_rate: I started with default 0.0001 and noticed the model's validation accuracy converges very slowly with relatively large epoches number (30), so I then tried 0.001 and 0.01. With these values the accuracy converges very fast but it keeps jump back and forth around ~90%, then I believe they are too large for my model. After iteratively experimenting, I settled with 0.0008.
- I increased training epoches number from 10 to 30, so the model can take full advantage of training data, although it takes more time and computing power.
- I did not adjust the default batch_size=128.
- I added dropout after flatten layer with keep_prob = 0.8, and after fully_connected_2 laye with keep_prob = 0.7. I started with 0.5 and found it results in underfitting. Then I slowly increase them and settled with 0.8 and 0.7.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I fed both conv1 and conv2 outputs into flatten layer, so that the low level features (in CONV1) of data can be taken account into in later layers, as described in the paper in the notebook. 
A dropout layer randomly takes any feature out of consideration with `1-keep_prob`, so the model will learn to not give great weight on any particular feature, but make decision based on all information it gets, making it a better model.

[//]: # (If a well known architecture was chosen:)
[//]: # (* What architecture was chosen?)
[//]: # (* Why did you believe it would be relevant to the traffic sign application?)
[//]: # (* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?)
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and third image might be difficult to classify because converting from its original size to the required size 32x32 makes the sign too blur to recognize even for human eyes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      	| General caution  								| 
| Stop sign     		| Stop sign										|
| Wild animals crossing	| Speed limit (70km/h)							|
| Go straight or right	| Go straight or right			 				|
| Road work				| Road work										|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is not as good as the accuracy on the test set of 92.4%. I think this is mainly because of my poor selection of images. After converting to 32x32, even for human eyes it is hard hard to recognize the image 1 and 3, so I was not surprised my model did not get them right.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .577         			| General caution								| 
| .999     				| Stop sign										|
| .899					| Speed limit (70km/h)							|
| .982	      			| Go straight or right			 				|
| .999				    | Road work         							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



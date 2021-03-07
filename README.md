# EmotionAI

## PROJECT OVERVIEW

DataSet: https://drive.google.com/drive/folders/1eOllPG-4pfFXqCLJOcrr8I89miQK_yxb?usp=sharing

Aim of the project is to detect the emotion and facial keypoints of the people from the face image.

Source: https://www.kaggle.com/c/facial-keypoints-detection/overview

Source: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

To get the data set: https://drive.google.com/drive/folders/1BF1AMkfrgY6GLHVoElb844Ejp4Geyd4_?usp=sharing

##INTRODUCTION TO EMOTION AI

• Artificial Emotional Intelligence or Emotion AI is a branch of AI that allow computers to understand human non-verbal cues such as body language and facial expressions.
• Affectiva offers cutting edge emotion AI tech: https://www.affectiva.com/

##PROJECT OVERVIEW

• The aim of this project is to classify people’s emotions based on their face images.
• In this project we have collected more than 20000 facial images, with their associated facial expression labels and around 2000 images with their facial key-point annotations.

##PART 1. KEY FACIAL POINTS DETECTION

• In part1, we will create a deep learning model based on Convolutional Neural Network and Residual Blocks to predict facial key-points.
• dataset of x and y coordinates of 15 The consists facial key points.
• Input Images are 96 x 96 pixels.
• Images consist of only one color channel (gray-scale images).

##PART 2. FACIAL EXPRESSION(EMOTION) DETECTION

• The second model will classify people’s emotion.
• Data contains images that belong to 5 categories:

0 = Angry
1 = Disgust
2 = Sad
3 = Happy
4 = Surprise

##NEURON MATHEMATICAL MODEL

• The brain has over 100 billion neurons communicating through electrical & chemical signals. Neurons communicate with each other and help us see, think, and generate ideas.
• Human brain learns by creating connections among these neurons. ANNs are information processing models inspired by the human brain.
• The neuron collects signals from input channels named dendrites, processes information in its nucleus, and then generates an output in a long thin branch called axon.

y=f(x1w1+x2w2+x3w3+b)

###EXAMPLE

• Bias allows to shift the activation function curve up or down.
• Number of adjustable parameters = 4 (3 weights and 1 bias).
• Activation function “F”.

##MULTI-LAYER PERCEPTRON NETWORK

• Let’s connect multiple of these neurons in a multi-layer fashion.
• The more hidden layers, the more “deep” the network will get.

To Play with Neural Network methodology :- https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.66088&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

3 Types of  Neural Network in Deep Learning Atricle :- https://www.analyticsvidhya.com/blog/2020/02/cnn-vs-rnn-vs-mlp-analyzing-3-types-of-neural-networks-in-deep-learning/

2D Visualization : https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html

3D Visualization : https://www.cs.ryerson.ca/~aharley/vis/conv/

##DIVIDE DATA INTO TRAINING AND TESTING

• Data set is generally divided into 80% for training and 20% for testing.
• Sometimes, we might include cross validation dataset as well and then we divide it into 60%, 20%, 20% segments for training, validation, and testing, respectively (numbers may vary).

1. Training set: used for gradient calculation and weight update. 
2. Validation set: 
	o used for cross-validation to assess training quality as training proceeds.
	o Cross-validation is implemented to overcome over-fitting which occurs when algorithm focuses on training set details at cost of losing generalization ability.
3. Testing set: used for testing trained network.

##GRADIENT DESCENT

• Gradient descent is an optimization algorithm used to obtain the optimized network weight and bias values 
• It works by iteratively trying to minimize the cost function
• It works by calculating the gradient of the cost function and moving in the negative direction until the local/global minimum is achieved
• If the positive of the gradient is taken, local/global maximum is achieved
• The size of the steps taken are called the learning rate
• If learning rate increases, the area covered in the search space will increase so we might reach global minimum faster
• However, we can overshoot the target
• For small learning rates, training will take much longer to reach optimized weight values
 
##RESNET (RESIDUAL NETWORK)

• As CNNs grow deeper, vanishing gradient tend to occur which negatively impact network performance. 
• Vanishing gradient problem occurs when the gradient is back-propagated to earlier layers which results in a very small gradient.
• Residual Neural Network includes “skip connection” feature which enables training of 152 layers without vanishing gradient issues.
• Resnet works by adding “identity mappings” on top of CNN.
• ImageNet contains 11 million images and 11,000 categories.
• ImageNet is used to train ResNet deep network.

##DEFINITIONS AND KPIS

• A confusion matrix is used to describe the performance of a classifififification model:
o True positives (TP): cases when classifier predicted TRUE (they have the disease), and correct class was TRUE (patient has disease).
o True negatives (TN): cases when model predicted FALSE (no disease), and correct class was FALSE (patient do not have disease).
o False positives (FP) (Type I error): classifier predicted TRUE, but correct class was FALSE (patient did not have disease).
o False negatives (FN) (Type II error): classifier predicted FALSE (patient do not have disease), but they actually do have the disease
o Classification Accuracy = (TP+TN) / (TP + TN + FP + FN)
o Misclassification rate (Error Rate) = (FP + FN) / (TP + TN + FP + FN)
o Precision = TP/Total TRUE Predictions = TP/ (TP+FP) (When model predicted TRUE class, how often was it right?)
o Recall = TP/ Actual TRUE = TP/ (TP+FN) (when the class was actually TRUE, how often did the classifier get it right?)

##MODEL DEPLOYMENT USING TENSORFLOW SERVING:

• Let’s assume that we already trained our model and it is generating good results on the testing data.
• Now, we want to integrate our trained Tensorflow model into a web app and deploy the model in production level environment.
• The following objective can be obtained using TensorFlow Serving. TensorFlow Serving is a high-performance serving system for machine learning models, designed for production environments.
• With the help of TensorFlow Serving, we can easily deploy new algorithms to make predictions.
• In-order to serve the trained model using TensorFlow Serving, we need to save the model in the format that is suitable for serving using TensorFlow Serving.
• The model will have a version number and will be saved in a structured directory.
• After the model is saved, we can now use TensorFlow Serving to start making inference requests using a specific version of our trained model "servable".

RUNNING TENSORFLOW SERVING:

• There are some important parameters:
o rest_api_port: The port that you'll use for REST requests.
o model_name: You'll use this in the URL of REST requests. You can choose any name
o model_base_path: This is the path to the directory where you've saved your model.

• For more information regarding REST, check this out: https://www.codecademy.com/articles/what-is-rest
• REST is a revival of HTTP in which http commands have semantic meaning.

MAKING REQUEST IN TENSORFLOW SERVING:

• In-order to make prediction using TensorFlow Serving, we need to pass the inference requests (image data) as a JSON object.
• Then, we use python requests library to make a post request to the deployed model, by passing in the JSON object containing inference requests (image data).
• Finally, we get the prediction from the post request made to the deployed model and then use argmax function to find the predicted class.


# To get complete theory, formula and example, Refer the PDF.



---
layout: post
title: "Creating a Cost of Care prediction with TensorFlow"
tags: [healthcare, cloud]
---

I speak to a lot of people about AI. About the possibilities for healthcare, the limitations, the tools, and how the Cloud providers help us quickly build models. I decided to explore what it takes to build, train and publish a simple AI model to predict a patient's insurance charges. In this 5 minute video, I explore how to create a simple 'cost of care' prediction model and gain experience with [TensorFlow][1], Google's open source software library for machine learning with particular focus on training of deep neural networks. I decided to explore the steps that it takes to go from zero to a fully trained published model. In this 5 minute video I'll show you the steps I had to take to create a blueprint for further investigation.

<!--more-->

<br/>
<iframe width="560" height="315" src="https://www.youtube.com/embed/6Zinxztsy5c" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Hello World with TensorFlow
This adventure is very much a simple 'Hello World' example with TensorFlow and Python. I'll use the following open source tools and you can read more about each of them by following the links:
- [Docker][11] for the sandpit environment. I'm using version 3.0.3, but any recent version should be fine.
- [Jupyter Lab][12] for Python coding
- [Scikit-Learn][13] for data normalization
- [Matplotlib][14] and [Seaborn][15] for visualisation
- [TensorFlow][16] for linear regression machine learning algorithm
- [Flask][17] for publishing the API.

These libraries will all be installed into the Docker container, which will be a quick sandpit environment.  I will build the  model in four simple steps:
- Step 1 - Build a TensorFlow sandpit environment
- Step 2 - Setup the Jupyter notebook
- Step 3 - Build and train the machine learning model using linear regression
- Step 4 - Publish the prediction model through an API using Flask

## Using the example from GitHub
If you're following along, the only thing you'll need installed is Docker, and code can be downloaded from [GitHub][2]. On GitHub you'll see a number of helper commands to help things run smoothly.

- *build* contains the command to build the Docker container from the [Dockerfile][10].
- *run* contains the commands to run our container, and also display the logs.
- *cleanup* will stop and delete the Docker container. 
- *logs* will show the log files of the running container.

You can copy the contents of GitHub to your local machine by running the [git][3] command.
```
git clone https://github.com/fiveminutecloud/fmctensorflow.git
```
or you can simply download the [zip file][18].

## Step 1 - Create the sandpit environment
Now, there are a few ways you can get a sandpit environment.  You can use something like AWS SageMaker, Azure Machine Learning, or Google Colab, but for this example i'll quickly make my own local environment. This helps  understand a little more about what's going on, but is also means I can use data sets which I might not be comfortable with sharing on the Cloud.

My Dockerfile uses the standard python 3.7 base image from DockerHub, on which I layer Jupyter Lab, the TensorFlow machine learning libraries and the Flask libraries to publish the API.

``` docker
FROM python:3.7-slim
RUN apt-get update
RUN pip3 install scikit-learn==0.23.2
RUN pip3 install jupyterlab==2.2.9
RUN pip3 install matplotlib==3.3.3
RUN pip3 install seaborn==0.11.0
RUN pip3 install numpy==1.18.5
RUN pip3 install tensorflow==2.3.1
RUN pip3 install flask==1.1.2
CMD ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/notebooks"]
```

Build the Docker container using the command
``` bash
docker build -t fmctensorflow .
```
## Step 2 - Setup the Jupyter Notebook
Now I have a container image, I run it with the following command. This command creates an instance of the container, and opens two ports on the container. Port 8888 is opened so I can access Jupyter Notebooks through the web browser, and port 5000 is opened for the API which I'll create shortly.

``` bash
docker run -d -v $(pwd)/notebooks:/notebooks --name fmctensorflow -p 8888:8888 -p 5000:5000 fmctensorflow:latest
docker logs -f fmctensorflow
```
Within the logs, there is a line which looks something like the line below. It contains the access token for the running instance of Jupyter, so I need that safely noted down. Simply copy the whole line into the browser, to access the Notebook. I recommend using the last link which starts http://127.0.0.1.
```
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://046ef7394882:8888/?token=9e976aecfb63ee81434e0272b4e996c130833ee891c91434e0272b4eaea6304e976aea8
     or http://127.0.0.1:8888/?token=e976aecfb63ee81434e0272b4e996c130833ee891c91434e0272b4eaea6304e976aea8
```
Now in the web browser, I see *Launcher* and click the Python3 logo to create a new Notebook.  You can right-click the file created on the left to rename it. 
<br/><br/>
![](../../../assets/myimages/2021-01-12-17-39-42.png)

## Step 3 - Build and train the TensorFlow model
You can view the actual Notebook [here][4], but I'm going to summarize the eleven key steps below.

### a. Import the libraries
Here I import the libraries that were layered into the Docker container during step 1. You'll notice I'm using some libraries which I didn't explicitly include (eg. Pandas), but I can still use them because they are part of the base Python 3.7 container.

### b. Retrieve the data set for training
The dataset I use for training is a public dataset from [Kaggle][5] with a small sample of USA population medical insurance costs. It has 1338 records, with columns as follows:
<br/><br/>

| Feature  | Description                        | Type       |
|----------|------------------------------------|------------|
| Age      | Patient age                        | Numeric    |
| Sex      | Patient sex                        | Text       |
| BMI      | Patient body-mass index            | Numeric    |
| Children | Number of children                 | Numeric    |
| Smoker   | Is the patient a smoker?           | Text       |
| Region   | Patient's home region              | Text       |
| Charges  | Insurance costs                    | Numeric    |

### c. Visualise and explore the data with Matplotlib
I am interested to understand the relationship between age and insurance costs. The following chart shows the distribution. Here I can see at least three linear relationships, so in the next step I'd like to understand the features that influence the insurance costs.
<br/><br/>
![](../../../assets/myimages/2021-01-13-13-20-06.png)

### d. Map the textual values to numerical values
In order to understand the features that influence insurance costs, I need to delve deeper into the data. I'd like to look at creating a *correlation matrix* to show how the costs change, as the features change. For example, do insurance costs go up as BMI increases?  How do the costs vary by region? 
In order to do this analysis, I use a *Correlation* function, but it only works with numeric data.  Therefore, I create three mapping functions to map Sex, Smoker and Region to numeric values, so I configure the following mappings. 
<br/><br/>

| Feature  | Text Values                             | Numeric Values                   |
|----------|-----------------------------------------|----------------------------------|
| Sex      | male, female, undefined                 | 0, 1, -1                         |
| Smoker   | no, yes, undefined                      | 0, 1, -1                         |
| Region   | southwest,southeast,northwest,northeast, undefined | 1, 2, 3, 4, 0                      |

### e. Look at the *correlation* between data items with Pandas and SeaBorn
I use *Pandas* to first create the Correlation matrix, and then I use *SeaBorn* to visualise the matrix to see the hotspots. From the charts below, I see the factors influencing the insurance costs are smoker, and then age. 
<br/><br/>

|-------------------------------------|----------------------------------|
| ![](../../../assets/myimages/2021-01-13-13-41-01.png)   | ![](../../../assets/myimages/Sex.png) |

### f. Separate the Smokers from the Nonsmokers
Since Smokers is a 'yes/no' value, in the next step I split the source data into Smokers and Nonsmokers so I can see the different trends. In the image below I can see the split of smokers and nonsmokers. I can see at least 3 linear relationships. I can see two for smokers (in grey), and a good linear relationship for nonsmokers (in blue) but with a sizable set of anomalies. So, for the rest of this example I'll focus on the nonsmokers only. 
<br/><br/>
![](../../../assets/myimages/2021-01-13-13-48-36.png)

### g. Look at the correlation between features for only Nonsmokers
Now, I re-run the Correlation function for nonsmokers only, and discover that age (unsurprisingly!) becomes the most significant factor in insurance charges.
<br/><br/>
![](../../../assets/myimages/2021-01-13-13-54-06.png)

### h. Normalize the input values with Scikit-Learn
I now need to prepare the data for training.  It is best practice to normalize the input values, and I do this by creating a *scaler*.  The scaler is a function that condenses the input dataset (age) down to a unit range (between 0-1). If you don't do this, you may find the modeling fit [*explodes*][6]. You don't need to normalize the output values (costs).
<br/><br/>
![](../../../assets/myimages/2021-01-13-13-56-33.png)


### i. Split the data into a training and testing dataset
There is one last thing needed before training the model. In step 3b, I noted 1338 records in the data set, of which 1064 are nonsmokers. I want to use most of the data for training the model. But, I also want to hold back some data so I can test the model afterwards and compare the predicted result with the actual result already known.  This helps understand the accuracy of the model. I decide to train the model on 1010 (95%) nonsmokers, and hold back 54 records for testing later. 

### j. Train the model with TensorFlow
*Finally!* Eleven steps later, I can train the model with TensorFlow. TensorFlow works by creating *models* and *layers*. Models are made up of layers, and layers are the *functions* containing the mathematical magic. The power of TensorFlow allows you to build complex networks with multiple layers leading to very sophisticated prediction models. I will only create one [Keras][7] layer.  The model fitting process is essentially *random*.  And if you run the fitting process multiple times, you'll get different results, unless you set the [seed][8] for the random number generator. Fitting basically selects a random position, and tests the training data against it.  It uses *Optimizers* to determine the accuracy (or 'loss') of the prediction against the known outcome in the training set and adjusts the random position accordingly.  It's like guessing a number between 1 and 100. You guess 50, I say lower. You guess 25, I say higher. Eventually you guess correctly. Sometimes you get lucky and it only takes a few guesses, other times it takes longer.  Each guess is called an *Epoch*, and the more guesses you have the longer it takes, and the better the result. I'm using just 1000 epochs here. I'm only using 1064 records to train our model, and it works just fine locally. But what if you have millions, or billions of records. Well, TensorFlow scales crazy-well, and lots of the cloud providers, but especially [Google Cloud AI][9] support TensorFlow at scale. All this work allows me to run just three commands to train the model.

``` python
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x=x_train, y=y_train, epochs=1000,shuffle=False,verbose=0)
```

I wait a few minutes for the modelling to run, and what do I get for my trouble? I get an *in memory* model that can be used for predictions. But first, let's evaluate the model against the 54 records held back in step 3i. I need to check the model makes reasonable predictions.

### k. Evaluate the model with the remaining data set
In this step I evaluate the model against the 54 records I held back.  I already know the insurance costs for these patients, so let's see what the new prediction model gives. The green-dots show the actual costs recorded in the source data, but the red-crosses show the predicted costs. And it's not too bad! Clearly, the test data has some anomalies, but remember in step 3g I choose to only focus on the patients *age*.  I haven't accounted for other features such as number of children, or BMI which may explain these deviations.

You may also notice that the *age* axis is still normalized because my model was trained on normalized *age* data. In the final step 4,  you'll see how I publish the model as an API, and use the *Scaler* from step 3h to normalize the age for which I want the insurance cost prediction. 
<br/><br/>
![](../../../assets/myimages/2021-01-13-15-53-41.png)

## Step 4 - Deploy the model as an API with Flask
I use the in-memory model created above, and create a web API around it so it can be used by mobile and web apps. I use Flask to create the API infrastructure, with one simple GET endpoint *predict*, that takes *age* as a single input value. When the container was started above in step 2, I opened port 5000 so we can access the Flask API though the web browser. I can now access the insurance cost predictor through one simple endpoint.
```
http://localhost:5000/predict?age=45
```

Remember, I said above the model works on a *normalized* age value, so I simply use the same scaler from step 3h we created before to scale the age.  
``` python
scaler_input_array = np.array([[age,0]]) 
scaled_age = mm_scaler.transform(scaler_input_array)[0][0] 
cost_prediction = model.predict([scaled_age])[0][0]  
```

I then pass the *normalized* age to the model predictor and return the insurance cost in the HTTP response.

<br/><br/>
![](../../../assets/myimages/2021-01-12-18-32-30.png)


## If you run into problems
I like testing examples in an isolated container because it provides a nice clean workspace with known dependencies. If you're following along with this 'Hello World' example and you run into difficulties, you can simply hit *reset* and start again from a known state. The container can be restarted by running the *cleanup* commands:
``` bash
docker kill fmctensorflow
docker rm fmctensorflow
```
You can then restart the container using the commands in step 2.  

If you have any other difficulties, please comment below.

[1]: https://to.fiveminute.cloud/1r2xMz
[2]: https://to.fiveminute.cloud/ekxWeR
[3]: https://git-scm.com/downloads
[4]: https://github.com/fiveminutecloud/fmctensorflow/blob/main/notebooks/fmctensorflow.ipynb
[5]: https://www.kaggle.com/mirichoi0218/insurance/home
[6]: https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
[7]: https://keras.io
[8]: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
[9]: https://towardsdatascience.com/multi-worker-distributed-tensorflow-training-on-google-cloud-ai-platform-64b383341dd8
[10]: https://github.com/fiveminutecloud/fmctensorflow/blob/main/Dockerfile
[11]: https://www.docker.com/products/docker-desktop
[12]: https://jupyterlab.readthedocs.io/en/stable/index.html
[13]: https://scikit-learn.org/stable/
[14]: https://matplotlib.org
[15]: https://seaborn.pydata.org
[16]: https://www.tensorflow.org
[17]: https://flask.palletsprojects.com/en/1.1.x/
[18]: https://github.com/fiveminutecloud/fmctensorflow/archive/main.zip



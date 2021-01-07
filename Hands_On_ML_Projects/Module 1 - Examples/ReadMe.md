# **Introduction**
The purpose of this project is to provide a comprehensive and yet a hands-on examples in Machine Learning using Tensorflow and Python.

There are many things that we can learn from data. Neural nets are the most common technique as of this writing, but they produce “black boxes.” We’ll precede our neural net discussion with some techniques that produce understandable extractions. The first  is to construct a decision tree and then Suport Vector Machine from data.

Please see file __**Decision Tree (Hands-on).ipynb**__ for an example of Decision Tree implementation.

**Decision Trees** are a classifier in machine learning that allows us to make predictions based on previous data. They are like a series of sequential “if … then” statements you feed new data into to get a result.

Please see file __**SVM (Hands-on).ipynb**__ for an example of SVM implementation.

A **Support Vector Machine** (SVM) is another machine learning algorithm that is used to classify data. The point of SVM's are to try and find a line or hyperplane to divide a dimensional space which best classifies the data points. If we were trying to divide two classes A and B, we would try to best separate the two classes with a line. On one side of the line/hyperplane would be data from class A and on the other side would be from class B. This alogorithm is very useful in classifying because we must to calculate the best line or hyperplane once and any new data points can easily be classified just by seeing which side of the line they fall on. 

### **Installation**
This hands-on examples assumes that you have at least some level of understanding on how Decision Trees and SVM works. It also assumes that the user(s) of the code has Python >= 3.5 installed and have and knows how install packages/libraries using <span style='color:red'> **__pip__** </span> and/or <span style='color:red'> **__conda__** </span>in their machine.

First check if the below packages has been installed. For example, to see if you have numpy installed you can run numpy:

```
    **import numpy as np**
    **print(np.__version__)**
```

You can perform the above steps in making sure that the rest of the libraries

```
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install scikit-learn
    pip install time
```

# Note: Google Colab
If you intend to use Colab -- I will highly suggest you first create a Gmail account if you don't have one. Then clone github https://github.com/febikamBU/Hands-on-ML-Project-for-CS-767.git
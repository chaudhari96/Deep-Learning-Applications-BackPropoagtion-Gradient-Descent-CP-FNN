# Deep-Learning-Applications-BackPropoagtion-Gradient-Descent-CP-FNN

1. Backpropagation (0.5pt)
    Draw the computation graph for the following function: 1/(1+(𝑎^b+c^d)∗𝑒)^2. Compute the gradient of the function with respect it to its inputs at (a,b,c,d,e) =   (1,1,1,1,1)

2. Gradient Descent 
   a. Write a function to compute the mean squared error between a prediction and ground truthassuming both are numpy arrays (see python module numpy)
   b. Consider a model: y = mx + c , where the model parameter m = 1 and parameter c = 0 and x ∈ (0,1) . Plot the function using matplotlib.
   c. Generate example data by drawing N = 100 uniform values from the range in which x lies, andcompute the corresponding y to get {x,y) over i = 1 to N.
   d. Assuming that you do not know the model parameters, use backpropagation and gradientdescent updates to fi nd the model parameters (choose an appropriate   learning rate). The lossfunction will be the mean squared error.
   e. Plot the error in the estimates as a function of the number of iterations of gradient update.Change the learning rate and plot another curve on the previous plot.
   f. Do steps 3-5 when the model is y = m1x + m2x^2 + c and the true parameters are m1 = 0.5, m2 = 1and c = 1. And x ∈ (0, 1). Also, plot the ground truth function    (i.e., with the true values of m1, m2 andc). Compare and contrast the plot with the previous one.
   g. Do steps 3-5 when the model is y = tanh(m*x + c) and the true parameters are m = 1 and c = 2.And x ∈ (0, 2). Also, plot the ground truth function

3. ML Basics (0.5pt)
    Write a function to compute the (multiclass) logistic loss (also called thecross-entropy loss) given the parameters (W,b) of a linear model (as numpy arrays)    and anexample (x,y). Add an l1 regularization and an l2 regularization to the loss function

4. Classifi cation Pipeline (2.5pt)
   a. Generate data from Data_Linear_Classifi er.ipynb (see the examples folder in the course repo).
   b. Split the data into test and train (20%:80%).
   c. Build a linear classifi er assuming the multiclass logistic loss and l2 an regularization for theweights only. Report the prediction accuracy on the training data and the test data and showappropriate plots.
   d. Introduce a cross validation scheme and justify your choice of parameters. What is the validationaccuracy compare to the test accuracy.
   e. What is the sensitivity of the model's performance to different learning rates and the number ofgradient descent iterations. Describe via suitable plots.
   f. What is the sensitivity of the model's performance to different regularization parameter values.Find the best regularization parameter using an exhaustive search procedure. Describe your choicevia suitable plots. What is the performance difference between using regularization and noregularization?
   g. What is the sensitivity of the model's performance with respect to a different test train split (e.g.,50%:50%).

require(neuralnet)
require(MASS)
require(grid)

?infert

dim(infert)
  
?neuralnet

# linear.output = FALSE for classification 

nn = neuralnet(case ~ age+parity+induced+spontaneous,
               data = infert, 
               hidden = 2, #sqrt(num of inputs) good shoice
               err.fct = "ce",
               linear.output = FALSE)

summary(nn)
plot(nn)
nn$weights
nn$result.matrix
nn$covariate
infert$case

# checking misclassification training error
nn$net.result[[1]]
nn1 = ifelse(nn$net.result[[1]]>0.5,1,0)
nn1
misClassErr = mean(infert$case != nn1)
misClassErr

# using backpropagation
############################################
nn.bp = neuralnet(case ~ age+parity+induced+spontaneous,
                  data = infert, 
                  hidden = 2, #sqrt(num of inputs) good shoice
                  err.fct = "ce",
                  linear.output = FALSE,
                  algorithm = "backprop",
                  learningrate = 0.01)
nn1.bp = ifelse(nn.bp$net.result[[1]]>0.5,1,0)
misClassErr.bp = mean(infert$case != nn1.bp)
misClassErr.bp

# making preductions and cross validation
############################################

# dummy data to make predictions
newdata = matrix(c(22,1,0,0,
                   22,1,1,0,
                   22,1,0,1,
                   22,1,1,1),
                   byrow = TRUE, ncol = 4)

# make predictions
newdata.pred = compute(nn, covariate = newdata)

# look at results
newdata.pred$net.result

# Confidence intervals
##############################################
# on the weights....

ci = confidence.interval(nn, alpha = 0.05)
ci

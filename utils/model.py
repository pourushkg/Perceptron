import numpy as np 


class Perceptron:
  def __init__(self,eta,epochs):
    self.weights = np.random.randn(3)*1e-4  # small weight unit
    print(f"Initial weight before training :\n{self.weights}")
    self.eta=eta, # Learning rate
    self.epochs=epochs

  def activationFunction(self,inputs, weights):
    z=np.dot(inputs,weights) # z=w1*x1 + w2*x2 + w3*x3
    return np.where( z > 0, 1, 0)

  def fit(self,x,y):
    self.x=x
    self.y=y
    x_with_bias= np.c_[self.x, -np.ones((len(self.x),1))] ## for concatination
    print(f"x with bias: \n{x_with_bias}")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(x_with_bias,self.weights) ## forward propagation
      print(f"predicted value after forward pass:\n{y_hat}")
      self.error = self.y -y_hat
      print(f"error = \n{self.error}")
      self.weights = self.weights + self.eta*np.dot(x_with_bias.T,self.error)
      print(f"updated weights after epoch: {epoch}/{self.epochs} = {self.weights}") ## backward propagation
      print("##"*10)

  def predict(self,x):
    x_with_bias=np.c_[x,-np.ones((len(x),1))]
    return self.activationFunction(x_with_bias,self.weights)


  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss: {total_loss}")
    return total_loss
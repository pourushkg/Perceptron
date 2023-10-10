from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd 
import numpy as np 

def main(data,eta,epochs,modelName,plotName):
    df=pd.DataFrame(data)
    print(df)

    x,y=prepare_data(df)

    model = Perceptron(eta=ETA,epochs=EPOCHS)
    model.fit(x,y)
    _ = model.total_loss()

    save_model(model,filename=modelName)
    save_plot(df,plotName,model)

if __name__ == "__main__":
    OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
    }
    ETA=0.3 # Learning rate lies between 0 and 1
    EPOCHS=10

    main(data=OR,eta=ETA,epochs=EPOCHS,modelName="or.model",plotName="or.png")

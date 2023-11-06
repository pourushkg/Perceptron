from utils.model import Perceptron
from utils.common import read_config
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd 
import numpy as np 
import logging 
import os
import argparse


logging_str = "[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,
                    filemode="a")

def main(data,config_path):
    config = read_config(config_path)
    df=pd.DataFrame(data)
    logging.info(f"This is a actual dataframe{df}")

    x,y=prepare_data(df)
    eta=config["xor"]["eta"]
    epochs=config["xor"]["epochs"]
    model = Perceptron(eta,epochs)
    model.fit(x,y)
    _ = model.total_loss()
    modelName = config["xor"]["modelName"]
    plotName = config["xor"]["plotName"]
    save_model(model,filename=modelName)
    save_plot(df,plotName,model)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")

    parsed_args = args.parse_args()

    XOR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
    }

    try:
        logging.info(">>>>>>>> starting training >>>>>>>>>")
        main(data=XOR,config_path=parsed_args.config)
        logging.info("<<<<<<<< Training Done <<<<<<<<<<<")

    except Exception as e:
        logging.exception(e)
        raise e 

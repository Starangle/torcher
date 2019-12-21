import torch
from progressbar import progressbar as pb
import numpy as np
import os
import time

class Torcher():

    @classmethod
    def problist2list(self,problist):
        if isinstance(problist,list):
            return problist
        elif problist==None:
            return []
        else:
            return [problist,]
    @classmethod
    def write_log(self,filename,text):
        if filename is not None:
            # if file not exist, create it
            if not os.path.exists(filename):
                if not os.path.exists(os.path.split(filename)[0]):
                    os.makedirs(os.path.split(filename)[0])
            # write log
            with open(filename,'a+') as f:
                f.write(text+'\n')

        print(text)

    def __init__(self,model,loss,opti,metrics=None,transforms=None):
        if isinstance(model,str):
            model=torch.load(model)
        assert isinstance(model,torch.nn.Module)
        self.model=model
        self.loss=loss
        self.metrics=self.problist2list(metrics)
        self.metrics_name=[x.__name__ for x in self.metrics]
        self.optimizer=opti(self.model.parameters())
        self.transforms=self.problist2list(transforms)
    
    
    def fit(self,train_data,valid_data=None,model_path=None,epochs=1,log_file=None):
        assert isinstance(train_data,torch.utils.data.DataLoader)

        self.write_log(log_file,time.asctime())
        self.model.train()

        for epo in range(epochs):
            
            self.write_log(log_file,'Epoch {}/{}'.format(epo+1,epochs))
            history=[]
            for x,y in pb(train_data):
                self.optimizer.zero_grad()
                record=[]

                x,y=x.cuda(),y.cuda()

                # apply transforms
                with torch.no_grad():
                    for transform in self.transforms:
                        x=transform(x)

                pred=self.model(x)
                loss=self.loss(pred,y)
                loss.backward()
                self.optimizer.step()
                record.append(loss.item())

                # eval metrics
                for metric in self.metrics:
                    metric_value=metric(pred,y)
                    record.append(metric_value)

                history.append(record)

            # print loss and metrics on this epoch
            statics=list(map(np.mean,list(zip(*history))))
            loss_tip="train loss is {:.4f}".format(statics[0])
            metric_tips=list(map(", {} is {:.4f}".format,self.metrics_name,statics[1:]))
            tip="".join([loss_tip,*metric_tips])
            self.write_log(log_file,tip)

            # valid
            valid_history=[]
            with torch.no_grad():
                for x,y in valid_data:
                    valid_record=[]
                    x,y=x.cuda(),y.cuda()
                    
                    # apply transforms
                    for transform in self.transforms:
                        x=transform(x)

                    pred=self.model(x)
                    loss=self.loss(pred,y)
                    valid_record.append(loss.item())

                    for metric in self.metrics:
                        metric_value=metric(pred,y)
                        valid_record.append(metric_value)
                    valid_history.append(valid_record)

            valid_statics=list(map(np.mean,list(zip(*valid_history))))
            loss_tip="valid loss is {:.4f}".format(valid_statics[0])
            metric_tips=list(map(", {} is {:.4f}".format,self.metrics_name,valid_statics[1:]))
            tip="".join([loss_tip,*metric_tips])
            self.write_log(log_file,tip)

            if model_path:
                torch.save(self.model,model_path)
        
    def save(self,model_path):
        torch.save(self.model,model_path)


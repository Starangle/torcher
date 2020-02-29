import torch
from progressbar import progressbar as pb
import numpy as np
import os
import time
from .callback import *

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
                target_dir=os.path.split(filename)[0]
                if target_dir!='' and not os.path.exists(target_dir):
                    os.makedirs(os.path.split(filename)[0])
            # write log
            with open(filename,'a+') as f:
                f.write(text+'\n')

        print(text)

    def __init__(self,model,loss,opti,metrics=None,transform=None):
        if isinstance(model,str):
            model=torch.load(model)
        assert isinstance(model,torch.nn.Module)
        self.model=model
        self.loss=loss
        self.metrics=self.problist2list(metrics)
        self.metrics_name=[x.__name__ for x in self.metrics]
        self.optimizer=opti(self.model.parameters())
        self.transform=transform
    
    def init_callbacks(self,callbacks):
        for cbk in callbacks:
            if isinstance(cbk,LearningRateDecay):
                cbk.associate(self.optimizer)
    
    def fit(self,train_data,valid_data=None,model_path=None,epochs=1,log_file=None,callbacks=[]):
        assert isinstance(train_data,torch.utils.data.DataLoader)

        self.write_log(log_file,time.asctime())
        self.init_callbacks(callbacks)

        for epo in range(epochs):

            for cbk in callbacks: #TODO: the parameters will be determined later
                if hasattr(cbk,'on_epoch_begin'):
                    cbk.on_epoch_begin()
            
            self.write_log(log_file,'Epoch {}/{}'.format(epo+1,epochs))
            self.model.train()
            history=[]
            for x,y in pb(train_data):
                if x is None:
                    continue
                self.optimizer.zero_grad()
                record=[]
                if self.transform:
                    with torch.no_grad():
                        x=self.transform(x)
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

            if isinstance(valid_data,torch.utils.data.DataLoader):
                # valid
                self.model.eval()
                valid_history=[]
                with torch.no_grad():
                    for x,y in valid_data:
                        if x is None:
                            continue
                        valid_record=[]
                        if self.transform:
                            x=self.transform(x)

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

            if True not in list(map(lambda x:isinstance(x,Checkpoint),callbacks)):
                if model_path:
                    torch.save(self.model,model_path)

            for cbk in callbacks:
                if hasattr(cbk,'on_epoch_end'):
                    if isinstance(cbk,LearningRateDecay):
                        cbk.on_epoch_end(epo,valid_statics[0])
                    elif isinstance(cbk,Checkpoint):
                        cbk.on_epoch_end(epo,valid_statics[0],valid_statics[1:],self.model)
                    else:
                        cbk.on_epoch_end()
        
    def save(self,model_path):
        torch.save(self.model,model_path)

    def load(self,model_path):
        self.model=torch.load(model_path)

    def eval(self,data,model_path):
        # valid
        self.load(model_path)
        self.model.eval()
        eval_history=[]
        with torch.no_grad():
            for x,y in data:
                if x is None:
                    continue
                eval_record=[]
                if self.transform:
                    x=self.transform(x)

                pred=self.model(x)
                loss=self.loss(pred,y)
                eval_record.append(loss.item())

                for metric in self.metrics:
                    metric_value=metric(pred,y)
                    eval_record.append(metric_value)
                eval_history.append(eval_record)

        eval_statics=list(map(np.mean,list(zip(*eval_history))))
        loss_tip="valid loss is {:.4f}".format(eval_statics[0])
        metric_tips=list(map(", {} is {:.4f}".format,self.metrics_name,eval_statics[1:]))
        tip="".join([loss_tip,*metric_tips])
        print(tip)



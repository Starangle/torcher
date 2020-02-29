import torch
class LearningRateDecay:
    def __init__(self,method='based_on_loss',decay=0.5,freq=3):
        assert method in ['based_on_loss','based_on_epoch']
        self.method=method
        self.decay=decay
        self.freq=freq
        self.best=[None,0]

    def associate(self,optimizer):
        self.optimizer=optimizer

    def update(self,):
        for p in self.optimizer.param_groups:
            p['lr'] *= self.decay

    def on_epoch_end(self,epoch,val_loss):
        if self.method=='based_on_loss':
            if self.best[0]==None or self.best[0]>val_loss:
                self.best=[val_loss,0]
            else:
                self.best[1]+=1
            if self.best[1]>=self.freq:
                self.update()
                self.best[1]=0

        if self.method=='based_on_epoch':
            if epoch%self.freq==0:
                self.update()

class Checkpoint:
    def __init__(self,name,mode='loss',metric_id=None):
        self.name=name
        self.mode=mode
        if mode=='metric':
            assert metric_id is not None
            self.id=metric_id
        self.best=None

    def on_epoch_end(self,epoch,val_loss,metrics,model):
        if self.mode=='loss':
            if self.best==None or self.best>val_loss:
                self.best=val_loss
                print('save on val loss {}'.format(val_loss))
                torch.save(model,self.name)
        if self.mode=='metric':
            if self.best==None or self.best<metrics[self.id]:
                self.best=metrics[self.id]
                print('save on val metric {}'.format(metrics[self.id]))
                torch.save(model,self.name)

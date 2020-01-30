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

            


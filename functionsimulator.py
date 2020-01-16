import torch
import random


#结果产生器
def resultgenerator(x):
    return  3*x +6*x+3
    
#函数模拟器
class model:
    def __init__(self):
        self.degree=2
        self.w=[random.random() for i in range(self.degree)]
 
    def forward(self,x):
        result=0    
        self.tmpx=[]
        for i in range(self.degree):
            tmpx=1
            for j in range(i):
                tmpx=tmpx*x
            self.tmpx.append(tmpx)                
            result=result+self.w[i]*tmpx
        return result
    def backward(self,loss):
        for i in range(self.degree):
            self.w[i]=self.w[i]-0.01*loss*self.tmpx[i]


m=model()
for i in range(100000000):

    x=random.random()
    y=m.forward(x)
    t=resultgenerator(x)
    loss=(y-t) 
    m.backward(loss)
    t=reversed(m.w)
    if i%100000==0:
        for i in t:
            print(i,end=" ")
        print(loss)
 
    

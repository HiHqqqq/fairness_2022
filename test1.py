import torch
out=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
a=torch.tensor([0,0,1,1])
a=a.unsqueeze(1)
b=1-a
#b=b.unsqueeze(0)
print(a,b)
sensitive=torch.cat((a,b),dim=1)
mask=sensitive==1
print(sensitive)

a=torch.masked_select(out,mask)
print(a)
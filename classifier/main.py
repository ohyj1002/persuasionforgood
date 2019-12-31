from torch import nn
from torch.autograd import Variable
from torch import optim
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from model import *
from tqdm import tqdm
from util import *
import json
import time
import matplotlib
import matplotlib.pyplot as plt
import copy
import os
import sys
plt.switch_backend('agg')

if __name__ == '__main__':
    n_label=11
    run_times=1    
    epochs=20

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    from charModel.encoder import Model
    char_model = Model()
    
    label_name=['Have','example',"logical",'PerStory','Foot',"Credibility","P-inqu","Emotional",'none',"t-inqu","info"]

    
    Model=RCNN_abla
    
    data_path = "./data/"
    best_f1=0

    train_accs = []
    test_accs = []

    
    acc_dict={}
    prec_dict={}
    cm_dict={}


    feat_list={'add_char':True,'add_turn':True,'add_senti':True,'add_his':True}
    feat='all'
    his_mode='rnn'


    acc_dict[feat]=[]
    prec_dict[feat]=[]
    cm_res=None
    for k in range(5):
        for i in range(run_times):

            train_iter,valid_iter, test_iter,TEXT, net,weight = load_data_model(model=Model, use_gpu=True,abla_list=feat_list,num_classes= n_label,f_path=data_path,char_model=char_model,his_mode=his_mode,cnt=k)
            net.optimizer = optim.Adam(params=net.parameters(), lr=1e-3)
            net.lr_scheduler = optim.lr_scheduler.StepLR(net.optimizer, step_size=100, gamma=0.95)
            net.loss_func =nn.CrossEntropyLoss() 
        

          
            best_f1=0

            train_accs = []
            test_accs = []


            

            for epoch in range(epochs):
                for phase in ('train', 'val'):
                    accs=AverageMeter()
                    losses= AverageMeter()
                    recalls=AverageMeter()
                    f1s=AverageMeter()
                    if phase == 'train':
                        net.train(True)
                        phrase_iter=train_iter
                        net.lr_scheduler.step()

                    else:
                        net.eval()
                        print("running valid.....")
                        phrase_iter=valid_iter
                    end = time.time()
                    for l in tqdm(phrase_iter): 

                        (xs,x_len),turn,(hs,hs_len),ys,senti,his_stem,char_embed,index=l

                        x_len=x_len.cuda().float().view(-1,1)
                        hs_len=hs_len.cuda().float().view(-1,1)
                        senti=torch.stack(senti,dim=1)


                        net.optimizer.zero_grad() #clear the gradient 
                        logits= net(xs,turn,hs,char_embed=char_embed,his_stem=his_stem,sentiment=senti)

                        loss= net.loss_func(logits, ys.data.long())
                


                        acc,f1=eval_(logits, labels=ys.data.long())


                        if phase == 'train':

                            loss.backward()
                            clip_grad_norm_(net.parameters(), 10)
                            net.optimizer.step()
                            train_accs.append(acc)

                        nsample = xs.size(0)
                        accs.update(acc, nsample)


                        f1s.update(f1, nsample)
                        
                        losses.update(loss.item(), nsample)


                    elapsed_time = time.time() - end


                    print('[{}]\tEpoch: {}/{}\tAcc: {:.2%}\tF1: {:.2%}\tLoss: {:.3f}\tTime: {:.3f}'.format(
                    phase, epoch+1, epochs,  accs.avg,f1s.avg,losses.avg, elapsed_time))
                    
                    

                

                    if phase == 'val' and f1s.avg > best_f1:
                        best_f1 = f1s.avg

                        best_epoch = epoch
                        best_model_state = net.state_dict()
                        preds=None
                        targets=None
                 
                        
                        test_accs=AverageMeter()
                        test_f1s=AverageMeter()
                        y_true=None
                        y_pred=None

                        for l in tqdm(test_iter):  
                            net.eval()
                            (xs,x_len),turn,(hs,hs_len),ys,senti,his_stem,char_embed,index=l
                            x_len=x_len.cuda().float().view(-1,1)
                            hs_len=hs_len.cuda().float().view(-1,1)
                            senti=torch.stack(senti,dim=1)
                            net.optimizer.zero_grad() #clear the gradient 




                            logits=net(xs,turn,hs,char_embed=char_embed,his_stem=his_stem,sentiment=senti)

                            output=logits
                            l_n=logits.data.cpu().numpy()
                            nsample = xs.size(0)

                            acc,f1=eval_(output, labels=ys.data.long())
                            _, predicted = torch.max(logits.cpu().data, 1)
                            test_accs.update(acc, nsample)
                            
                            test_f1s.update(f1, nsample)
                            if y_true is None:
                                y_true=ys.data.cpu().numpy()
                                y_pred=l_n.argmax(axis=1)
                            else:
                                y_true=np.hstack([y_true,ys.data.cpu().numpy()])
                                y_pred=np.hstack([y_pred,l_n.argmax(axis=1)])

                        print('[test]\tEpoch: {}/{}\tAcc: {:.2%}\tF1: {:.2%}\tTime: {:.3f}'.format(
                          epoch+1, epochs,  test_accs.avg,test_f1s.avg, elapsed_time))
                        from sklearn.metrics import confusion_matrix
                        cm=confusion_matrix(y_true, y_pred)
                        print(cm.shape)
                        print_cm(cm, label_name)

            print('[Info] best valid acc: {:.2%} at {}th epoch'.format(best_f1, best_epoch))
            torch.save(best_model_state, 'best_model_state_er.pkl')
            print('Test Acc: {:.2%}\tF1: {:.2%}\t'.format(
                           test_accs.avg,test_f1s.avg))
            acc_dict[feat].append(test_accs.avg)
            prec_dict[feat].append(test_f1s.avg)
            if cm_res is None:
                cm_res=cm
            else:
                cm_res+=cm
    cm_dict[feat]=cm_res.tolist()
    res_dict={}
    res_dict['acc']=acc_dict
    res_dict['f1']=prec_dict
    res_dict['cm']=cm_dict
    with open('final.json','w') as f:
        json.dump(res_dict,f)

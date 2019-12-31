import pandas as pd
import numpy as np
import numpy as np
import re
import logging
import re
import sys
import spacy
from sklearn.metrics import f1_score,precision_score,classification_report,hamming_loss,recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

spacy_en = spacy.load('en')
import torch
import torch.nn as nn
from torch.nn import init
from torchtext import vocab
from torchtext import data
from torchtext.data import Iterator, BucketIterator
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if cm[i, j] != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def l2_matrix_norm(m):
    """
    Frobenius norm calculation

    Args:
       m: {Variable} ||AAT - I||

    Returns:
        regularized value

    """
    sum_res=torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5)
    sum_res=sum_res.float().cuda() 
    return sum_res


def eval_(logits, labels,binary=False):
    """
    calculate the accuracy
    :param logits: Variable [batch_size, ]
    :param labels: Variable [batch_size]
    :return:
    """
    if binary is False:
        _, predicted = torch.max(logits.data, 1)
        acc=(predicted == labels).sum().item()/labels.size(0)
        
        f1 = f1_score(labels.cpu(),predicted.cpu(), average='macro')
        
        return acc,f1
    else:
        l=torch.ones(logits.size()).cuda()
        
        l[logits <= 0] = 0

        return (l== labels).sum().item()/labels.size(0)
    
    
    
def tokenizer(text): # create a tokenizer function
    text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']
    tokenized_text = []
    auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s","'m","'re"]
    punctuation='.,/\\-*&#\'\"'
    num=set('1234567890')
    stop_words=['a','the','of','in','on','to']
    for token in text:
        if token == "n't":
            tmp = 'not'
        elif token == 'US':
            tmp = 'America'
        elif token == "'ll":
            tmp = 'will'
        elif token == "'m":
            tmp = 'am'
        elif token == "'s":
            tmp = 'is'
        elif len(set(token) & num)>0:
            continue
        elif token == "'re":
            tmp = 'are'

        elif token in punctuation:
            continue
        elif token in stop_words:
            continue
        else:
            tmp = token
        tmp=tmp.lower()
        tokenized_text.append(tmp)
    return tokenized_text



class BatchWrapper:
    def __init__(self, b_iter, x_var, t_var, h_var, y_var, neg_var,neu_var,pos_var,stem_var,char_var,index):
        self.b_iter, self.x_var,self.t_var, self.h_var, self.y_var,self.neg_var,self.neu_var,self.pos_var,self.stem_var,self.char_var,self.index = b_iter, x_var, t_var, h_var,y_var ,neg_var,neu_var,pos_var,stem_var,char_var,index

    def __iter__(self):
        for batch in self.b_iter:
            
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            t= getattr(batch, self.t_var)
            h= getattr(batch, self.h_var)
            neg= getattr(batch, self.neg_var)
            neu= getattr(batch, self.neu_var)
            pos= getattr(batch, self.pos_var)
            stem= getattr(batch, self.stem_var)
            char= getattr(batch, self.char_var)
            index= getattr(batch, self.index)
            
            
            

            if self.y_var is not None:
                y=getattr(batch, self.y_var)
                
            else:
                y = torch.zeros((1))
            

            
            yield (x,t,h,y,(neg,neu,pos),stem,char,index)
            
    
    def __len__(self):
        return len(self.b_iter)

    
    
    
class custom_Field(data.Field):
    def __init__(self,model, **kwargs):
        self.model = model
        super(custom_Field, self).__init__(**kwargs)
    
    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        text_feat = self.model.transform(arr)
        text_feat = torch.from_numpy(text_feat).float().cuda()
        return text_feat
        
    
def load_data_model(model,n_layers=2,drop=0.5,use_gpu=False,num_classes= 6,f_path='./data/',abla_list=None,his_mode='rnn',char_model=None,cnt=0):    
    df = pd.read_csv(f_path+'train'+str(cnt)+'.csv')
    label_name='label'
    print(df.head(2))
    datafileds=[]
    df.label.value_counts()
    weight=np.array([df.shape[0]/v for v in df.label.value_counts().sort_index().tolist()])
    weight = np.log(weight)
    # Fit tf-idf vector
    from sklearn.feature_extraction.text import TfidfVectorizer


    df_tf_idf=df
    corpus = df_tf_idf.his_stem.tolist()
    vectorizer = TfidfVectorizer(max_features= 1000)
    vectorizer = vectorizer.fit(corpus)

    # Define field
    TEXT =data.Field(sequential=True,tokenize=tokenizer, use_vocab=True, lower=True,eos_token='<EOS>',batch_first=True, truncate_first=True,include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    TURN = data.Field(sequential=False, use_vocab=False)
    INDEX= data.Field(sequential=False, use_vocab=False)
    HIS = data.Field(sequential=True,tokenize=tokenizer, use_vocab=True, lower=True,eos_token='<EOS>',batch_first=True, truncate_first=True,include_lengths=True)
    
    
    HIS_STEM = data.Field(sequential=False,truncate_first=True)
    def numer(arr, device):
        arr=vectorizer.transform(arr).toarray()
        var = torch.tensor(arr, dtype=torch.float,device=device)
        return var
    HIS_STEM.numericalize=numer
    if char_model is None:
        CHAR_FEAT =data.Field(sequential=True,tokenize=tokenizer, use_vocab=True, lower=True,eos_token='<EOS>',batch_first=True, truncate_first=True,include_lengths=True)
    else:
        CHAR_FEAT = custom_Field(model=char_model,sequential=False, use_vocab=False)
        
    NEG = data.Field(sequential=False, dtype=torch.float, use_vocab=False,postprocessing=data.Pipeline(lambda x,y: float(x)))
    NEU = data.Field(sequential=False, dtype=torch.float, use_vocab=False,postprocessing=data.Pipeline(lambda x,y: float(x)))
    POS = data.Field(sequential=False, dtype=torch.float, use_vocab=False,postprocessing=data.Pipeline(lambda x,y: float(x)))

   
    for col in df.columns.tolist():
        if col =='Unit':
            datafileds.append((col,TEXT))
        elif col =='Unit_char':
            datafileds.append((col,CHAR_FEAT))
        elif col == 'Index':
            datafileds.append((col,INDEX))
            
        elif col ==label_name:
            datafileds.append((col,LABEL))
        elif col=='history':
            
            datafileds.append((col,HIS))
            
        elif col=='Turn':
            datafileds.append((col,TURN))
        elif col == "neg":
            datafileds.append((col,NEG))
        elif col == "neu":
            datafileds.append((col,NEU))
        elif col == "pos":
            datafileds.append((col,POS))
        elif col =="his_stem":
            datafileds.append((col,HIS_STEM))
        else:
            datafileds.append((col,None))

    # train,valid=dataset.split(split_ratio=0.8,random_state=np.random.seed(42))
    train,valid= data.TabularDataset.splits(   
           format='csv',
           skip_header=True,
           path=f_path,
           train='train'+str(cnt)+'.csv',
           validation= 'test'+str(cnt)+'.csv',
           fields=datafileds,
            )
    test = data.TabularDataset(
        path=f_path+'test'+str(cnt)+'.csv',    
           format='csv',
           skip_header=True,
           fields=datafileds,
            )

    # using the training corpus to create the vocabulary
    HIS.build_vocab(train,valid,test)
    HIS.vocab.load_vectors(vectors='fasttext.en.300d')
    TEXT.build_vocab(train,valid,test)#, vectors=vectors, max_size=300000)
    TEXT.vocab=HIS.vocab
    
    CHAR_FEAT.vocab=HIS.vocab
    print('num of tokens', len(TEXT.vocab.itos))
    print('num of tokens', len(HIS.vocab.itos))
    
    print(TEXT.vocab.freqs.most_common(5))
    print(HIS.vocab.freqs.most_common(5))
    
    print('len(train)', len(train))
    print('len(test)', len(test))
    

    train_iter = data.Iterator(dataset=train, batch_size=32 ,train=True, sort_key=lambda x: len(x.Unit), sort_within_batch=False,repeat=False,device=torch.device('cuda:0') if use_gpu else -1)
    
    val_iter = data.Iterator(dataset=valid, batch_size=256, train=False, sort_key=lambda x: len(x.Unit), sort_within_batch=False,repeat=False,device=torch.device('cuda:0') if use_gpu else -1)
    
    test_iter = data.Iterator(dataset=test, batch_size=256, train=False, sort_key=lambda x: len(x.Unit), sort_within_batch=False,repeat=False,device=torch.device('cuda:0') if use_gpu else -1)


    num_tokens = len(HIS.vocab.itos)


    hidden_dim=200
    
    print("No .class",num_classes)
    nets = model(vocab_size=num_tokens, embedding=TEXT.vocab.vectors, hidden_dim=hidden_dim, output_dim=num_classes, n_layers=n_layers, bidirectional=True, dropout=drop,his_mode=his_mode,**abla_list)
    
    
    train_iter = BatchWrapper(train_iter,"Unit",'Turn',"history",label_name,"neg","neu","pos",'his_stem','Unit_char','Index')
    valid_iter = BatchWrapper(val_iter, "Unit",'Turn',"history",label_name,"neg","neu","pos",'his_stem','Unit_char','Index')
    test_iter = BatchWrapper(test_iter, "Unit",'Turn',"history",label_name,"neg","neu","pos",'his_stem','Unit_char','Index')
  
    weight=torch.from_numpy(weight).float().cuda()
    
    if use_gpu:
        cuda1 = torch.device('cuda:0')
        nets.cuda(device=cuda1)
        return train_iter, valid_iter,test_iter,TEXT,nets,weight
    else:
        return train_iter, valid_iter,test_iter,TEXT,nets,weight


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=8, alpha=None, gamma=5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
    
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss    

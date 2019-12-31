import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import random

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding,self.embedding_dim=self._load_embeddings(vocab_size,use_pretrained_embeddings=True,embeddings=embedding)
        #self.hidden_state = self._init_hidden()
        self.rnn = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers,batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bool_fc =  nn.Linear(hidden_dim*2, 1)
    def _load_embeddings(self,vocab_size,emb_dim=None,use_pretrained_embeddings=False,embeddings=None):
        """Load the embeddings based on flag"""
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
                
        return word_embeddings,emb_dim
       
    def forward(self, x,*args,**kwargs):
        
        #x = [sent len, batch size]
        #embedded = [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(x))
        #output = [sent len, batch size, hid dim * num directions]
        #(hidden,cell) = ([num layers * num directions, batch size, hid dim]*2)
        
        outputs, (hidden,cell) = self.rnn(embedded)
        #hidden [batch size, hid. dim * num directions]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
#         sf=nn.Softmax(dim=1)
        output = self.fc(hidden)         
        bool_l=self.bool_fc(hidden)    
        return output,bool_l,bool_l
        return output
    
class RCNN_abla(LSTM):
    def __init__(self, vocab_size, embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout,batch_size=64, tfidf_dim=1000,user_embed_dim=10,
                 his_mode='rnn',add_persona=False,**kwargs):

        super(RCNN_abla, self).__init__(vocab_size, embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.dropout=nn.Dropout(dropout)
        self.batch_size = batch_size
        self.output_size = output_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        char_dim=4096
        turn_emb_dim = 10
        self.his_mode = his_mode
        self.add_senti= kwargs.get('add_senti',False)
        self.add_turn= kwargs.get('add_turn',False)
        self.add_char= kwargs.get('add_char',False)
        self.add_his= kwargs.get('add_his',False)
        print('----Ablation study----','senti',self.add_senti,'turn',self.add_turn,'char',self.add_char,'his',self.add_his)
        self.word_embeddings,embedding_dim=self._load_embeddings(vocab_size,use_pretrained_embeddings=True,embeddings=embedding)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout,batch_first=True,bidirectional=bidirectional)
        self.W2 = nn.Linear(2*hidden_dim+embedding_dim, hidden_dim)
        self.utt_fc = nn.Linear(hidden_dim,50)
        self.turn_embeddings=torch.nn.Embedding(30,turn_emb_dim) 
        
        # Mode for adding history feature
        if self.add_his is True:
            if self.his_mode== 'cnn':
                padding=(1,0)
                stride=2
                kernel_heights=[3,4,5]
                in_channels=1
                out_channels=50
                
                self.his_conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_dim), stride, padding)
                self.his_conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_dim), stride, padding)
                self.his_conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_dim), stride, padding)
                his_in_dim=len(kernel_heights)*out_channels
                his_dim = 50
                self.his_fc = nn.Linear(his_in_dim,his_dim)
                
            elif self.his_mode== 'rnn':
                self.his_rnn = nn.LSTM(self.embedding_dim, hidden_dim,  bidirectional=bidirectional, dropout=dropout,batch_first=True)
                his_dim=0
            elif self.his_mode== 'mean':
                his_dim=50
                self.his_fc = nn.Linear(embedding_dim,50)
            elif self.his_mode== 'tfidf':
                his_dim=50
                self.his_fc = nn.Linear(tfidf_dim,his_dim)
            elif self.his_mode =='attn':
                his_dim=50
                self.his_rnn = nn.LSTM(self.embedding_dim, hidden_dim,  bidirectional=bidirectional, dropout=dropout,batch_first=True)
                self.his_fc = nn.Linear(2*hidden_dim,50)
                # Add self attention
                
        else: 
            his_dim = 0  
            
        # Final output with all concatenated features
        final_out_dim = 50
        self.label_char = nn.Linear(char_dim, 50)
        
        self.attn_dim=150
        self.hops=1
        self.W_s1 = nn.Linear(2*hidden_dim, self.attn_dim)
        self.W_attn = nn.Linear(self.attn_dim, self.hops)
        if self.add_char is True:
            final_out_dim += 50
        if self.add_senti is True:
            final_out_dim += 3
        if self.add_turn is True:
            final_out_dim += turn_emb_dim
        self.label_final = nn.Linear(final_out_dim+his_dim,self.output_size)

    def attention_weight(self, lstm_output):
        attn_weight_matrix = self.W_attn(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        return attn_weight_matrix     

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out
        

    def forward(self, input_sentence,turn,his=None,char_embed=None,his_stem=None,sentiment=None):

        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        turn: turn index of current sentence in the dialogues
        his: context sentences
        char_embed: char embedding for sentences
        his_stem: stemming sentence
        sentiment: sentiment feature
        
        Returns
        -------
        
        output: shape = (batch_size, output_size)
        
        """
        
        """
        
        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
        of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
        its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
        state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_dim
        vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
        dimension tensor of size (batch_size, hidden_dim) and finally we map this to the output layer.
        """
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences, embedding_dim)
        input=self.dropout(input)
        h_0 = Variable(torch.zeros(2, input.size(1), self.hidden_dim).cuda()) # Initial hidden state of the LSTM
        c_0 = Variable(torch.zeros(2, input.size(1), self.hidden_dim).cuda()) # Initial cell state of the LSTM
    

        
        his_out = None
        if self.add_his is True:
            
            if self.his_mode=='tfidf': # TF-IDF
                his_out=F.relu(self.his_fc(his_stem))
            elif self.his_mode == 'mean': # Mean embed
                his_embed = self.word_embeddings(his)
                his_out = torch.mean(his_embed,dim=1)
                his_out=F.relu(self.his_fc(his_out))
            elif self.his_mode == 'cnn': # CNN model
                his_embed = self.word_embeddings(his)
                his_embed = his_embed.unsqueeze(1)
                max_out1 = self.conv_block(his_embed, self.his_conv1)
                max_out2 = self.conv_block(his_embed, self.his_conv2)
                max_out3 = self.conv_block(his_embed, self.his_conv3)
                his_out = torch.cat((max_out1, max_out2, max_out3), 1)
                his_out= F.relu(self.his_fc(his_out))
            elif self.his_mode == 'rnn':  # RNN model
                his_embed=self.embedding(his)
                his_embed = his_embed

                output, (hidden,cell)= self.his_rnn(his_embed)
                h_0 = hidden
                c_0 = Variable(torch.zeros(2, input_sentence.size(0), self.hidden_dim)).cuda() 
            elif self.his_mode =='attn': # RNN with attention
                his_embed=self.embedding(his)
                his_embed = his_embed

                output, (hidden,cell)= self.his_rnn(his_embed)
                weight=self.attention_weight(output )
                hidden_matrix = torch.bmm(weight, output)
                hidden_matrix=hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
                # First concate history
                his_out = F.relu(self.his_fc(hidden_matrix))

                
                
               
        if self.add_his is True and self.his_mode == 'rnn':
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0,c_0))
            
            
        else:
            output, (final_hidden_state, final_cell_state) = self.lstm(input)
        
        final_encoding = torch.cat((output, input), 2)
        y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_dim)
        y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_dim, num_sequences)
        y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_dim, 1)
        y = y.squeeze(2)
        logits = F.relu(self.utt_fc(y))
        
        if self.add_his is True and self.his_mode is not 'rnn':
            logits = torch.cat((logits,his_out), 1)       
        # Then after 100dim representation, concate with hand craft feature
        if self.add_turn is True:
            t_embed=self.turn_embeddings(turn)
            logits = torch.cat((logits,t_embed), 1)
            
        if self.add_char is True:
            char_emb=self.label_char(char_embed)
            logits = torch.cat((logits,char_emb), 1)
            
        if self.add_senti is True:
            logits = torch.cat((logits,sentiment), 1)

        output=self.label_final(logits)
        return output

    

    

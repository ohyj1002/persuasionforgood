#############################
#  requirement:
#
#  pip install vaderSentiment
#  pip install shorttext
#
#############################

import pandas as pd
from sklearn.model_selection import KFold
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from shorttext.utils import standard_text_preprocessor_1


def analyze(row):
    analyzer = SentimentIntensityAnalyzer()

    vs = analyzer.polarity_scores(row['Unit'])
    return pd.Series([vs['neg'], vs['neu'], vs['pos']])



df=pd.read_excel('../data/AnnotatedData/300_dialog.xlsx',index_col=[0])
df_all=pd.read_csv('../data/FullData/full_dialog.csv',index_col=[0])
df_all_ee=df_all[df_all.B4==1]
df_all_ee.loc[:,'B4']=0
df_his=df_all_ee
df_his.loc[:,'Turn']+=1
df_his=df_his.rename(columns={'Unit':'history'})


df_er=df[df.B4==0]

# merge persuader sentence with context (prevoious presuadee utterance)
df_er=pd.merge(df_er,df_his,how='left',on=['B2','B4','Turn'])
df_er.loc[(df_er.Turn==0),'history']='<start>'
# add sentiment feature
df_er[['neg', 'neu','pos']]=df_er.apply(analyze,axis=1)
preprocessor = standard_text_preprocessor_1()
# add history stemming result

df_er['his_stem']=df_er.apply(lambda x: preprocessor(x.history) ,axis=1)
df_er.loc[(df_er.his_stem==""),'his_stem']='none'

df_er['Unit_char']=df_er['Unit']
# add label field
label_set=set(df_er.er_label_1)
label_dict=dict(zip(label_set,range(len(label_set))))
df_er['label']=df_er['er_label_1'].apply(lambda x: label_dict[x])

df_er['Index']=range(df_er.shape[0])



#split data for cross validation
kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
cnt=0
for tr_idx,val_idx in kf.split(df_er):
    
    train = df_er.iloc[tr_idx]
    test =  df_er.iloc[val_idx]
    train.to_csv('./data/train'+str(cnt)+'.csv')
    test.to_csv('./data/test'+str(cnt)+'.csv')
    cnt+=1



# StackOverflow_TagsClassifier

    import numpy as np # linear algebra
    import pandas as pd # data processing
    import collections
   
   </br>
   
    ques = pd.read_csv('Questions.csv',encoding='iso-8859-1')
    ques.head(10)

  
      Id OwnerId	CreationDate	ClosedDate	              Score	Title                                             Body
      0	 80	      26.0	        2008-08-01T13:57:07Z	NaN	26	SQLStatement.execute() - multiple queries in o...	<p>I've written a database generation script i...
      1	 90	      58.0	        2008-08-01T14:41:24Z	2012-12-26T03:45:49Z	144	Good branching and merging tutorials for Torto...	<p>Are there any really good tutorials explain...
      2	 120	    83.0	        2008-08-01T15:50:08Z	NaN	21	ASP.NET Site Maps	<p>Has anyone got experience creating <strong>...
      3	 180	    2089740.0	    2008-08-01T18:42:19Z	NaN	53	Function for creating color wheels	<p>This is something I've pseudo-solved many t...
      4	 260	    91.0	        2008-08-01T23:22:08Z	NaN	49	Adding scripting functionality to .NET applica...	<p>I have a little game written in C#. It uses...
      5	 330	    63.0	        2008-08-02T02:51:36Z	NaN	29	Should I use nested classes in this case?	<p>I am working on a collection of classes use...
      6	 470	    71.0	        2008-08-02T15:11:47Z	2016-03-26T05:23:29Z	13	Homegrown consumption of web services	<p>I've been writing a few web services for a ...
      7	 580	    91.0	        2008-08-02T23:30:59Z	NaN	21	Deploying SQL Server Databases from Test to Live	<p>I wonder how you guys manage deployment of ...
      8	 650	    143.0	        2008-08-03T11:12:52Z	NaN	79	Automatically update version number	<p>I would like the version property of my app...
      9	 810	    233.0	        2008-08-03T20:35:01Z	NaN	9	Visual Studio Setup Project - Per User Registr...	<p>I'm trying to maintain a Setup Project in <...
      
</br>

    ques.drop(["OwnerUserId","CreationDate","ClosedDate","Score"], axis=1, inplace=True)
    ques.head(10)
</br>

    	Id	Title	                                            Body
    0	80	SQLStatement.execute() - multiple queries in o...	<p>I've written a database generation script i...
    1	90	Good branching and merging tutorials for Torto...	<p>Are there any really good tutorials explain...
    2	120	ASP.NET Site Maps	<p>Has anyone got experience creating <strong>...
    3	180	Function for creating color wheels	<p>This is something I've pseudo-solved many t...
    4	260	Adding scripting functionality to .NET applica...	<p>I have a little game written in C#. It uses...
    5	330	Should I use nested classes in this case?	<p>I am working on a collection of classes use...
    6	470	Homegrown consumption of web services	<p>I've been writing a few web services for a ...
    7	580	Deploying SQL Server Databases from Test to Live	<p>I wonder how you guys manage deployment of ...
    8	650	Automatically update version number	<p>I would like the version property of my app...
    9	810	Visual Studio Setup Project - Per User Registr...	<p>I'm trying to maintain a Setup Project in <...



</br>

    import re 

    def rem_html_tags(body):
        regex = re.compile('<.*?>')
        return re.sub(regex, '', body)
     
    ques['Body'] = ques['Body'].apply(rem_html_tags)
    ques.head()
    
    
 </br>
 
     df_tags = pd.read_csv('Tags.csv', encoding='iso-8859-1')
     df_tags.head()
     
  </br>
  
    import numpy as np # linear algebra
    import pandas as pd # data processing
    # Plotting Libs
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    %matplotlib inline

    import collections
    
 </br>
 
    def plot_tags(tagCount):
    
    x,y = zip(*tagCount)

    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
    colors = [colormap(i) for i in np.linspace(0, 0.8,50)]   

    area = [i/4000 for i in list(y)]   # 0 to 15 point radiuses
    plt.figure(figsize=(9,8))
    plt.ylabel("Number of question associations")
    for i in range(len(y)):
        plt.plot(i,y[i], marker='o', linestyle='',ms=area[i],label=x[i])

    plt.legend(numpoints=1)
    plt.show()
    
 </br>
 
    tagCount =  collections.Counter(list(df_tags['Tag'])).most_common(10)
    print(tagCount)
    plot_tags(tagCount)
    
 </br>
    
![image](https://user-images.githubusercontent.com/41589522/133224238-7e42dda6-d727-41b4-b2ea-6ae3553355be.png)
</br>

    top10=['javascript','java','c#','php','android','jquery','python','html','c++','ios']

</br>

    tag_top10= df_tags[df_tags.Tag.isin(top10)]
    print (tag_top10.shape)
    tag_top10.head()
    
</br>

    total=pd.merge(ques, top10_tags, on='Id')
    print(total.shape)
    total.head()

</br>

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from nltk import word_tokenize
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing import sequence
    from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, BatchNormalization, GRU ,concatenate
    from keras.models import Model
</br>

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(total.Tags)
    labels = multilabel_binarizer.classes_
    labels  
    #output:
    array(['android', 'c#', 'c++', 'html', 'ios', 'java', 'javascript',
       'jquery', 'php', 'python'], dtype=object)
 
 </br>
 
    train,test=train_test_split(total[:550000],test_size=0.25,random_state=24)
 
  </br>
  
    total=pd.merge(ques, top10_tags, on='Id')
    print(total.shape)
    total.head()
    
    X_train_t=train['Title']
    X_train_b=train['Body']
    y_train=multilabel_binarizer.transform(train['Tags'])
    X_test_t=test['Title']
    X_test_b=test['Body']
    y_test=multilabel_binarizer.transform(test['Tags'])

</br>

    import nltk
    nltk.download('punkt')


</br>

    sent_lens_t=[]
    for sent in train['Title']:
        sent_lens_t.append(len(word_tokenize(sent)))
    max(sent_lens_t)  
   
 </br>
 
    np.quantile(sent_lens_t,0.97)
    
 </br>
 
    max_len_t = 18
    tok = Tokenizer(char_level=False,split=' ')
    tok.fit_on_texts(X_train_t)
    sequences_train_t = tok.texts_to_sequences(X_train_t)
    
 </br>
 
    vocab_len_t=len(tok.index_word.keys())
    vocab_len_t
    
 </br>
  
    sequences_matrix_train_t = sequence.pad_sequences(sequences_train_t,maxlen=max_len_t)
    sequences_matrix_train_t
    
 </br>
 
    sequences_test_t = tok.texts_to_sequences(X_test_t)
    sequences_matrix_test_t = sequence.pad_sequences(sequences_test_t,maxlen=max_len_t)
    
  </br>
  
    sequences_matrix_train_t.shape,sequences_matrix_test_t.shape,y_train.shape,y_test.shape
    
  </br>
  
    sent_lens_b=[]
    for sent in train['Body']:
        sent_lens_b.append(len(word_tokenize(sent)))
    max(sent_lens_b)
    
  </br></br>
    
    vocab_len_b =len(tok.index_word.keys())
    vocab_len_b
    
  </br>
    
    sequences_matrix_train_b = sequence.pad_sequences(sequences_train_b,maxlen=max_len_b)
    sequences_matrix_train_b
    
  </br>
  
    sequences_test_b = tok.texts_to_sequences(X_test_b)
    sequences_matrix_test_b = sequence.pad_sequences(sequences_test_b,maxlen=max_len_b)
    
    
  </br>
  
    def RNN():
      # Title Only
      title_input = Input(name='title_input',shape=[max_len_t])
      title_Embed = Embedding(vocab_len_t+1,2000,input_length=max_len_t,mask_zero=True,name='title_Embed')(title_input)
      gru_out_t = GRU(300)(title_Embed)
      # auxiliary output to tune GRU weights smoothly 
      auxiliary_output = Dense(10, activation='sigmoid', name='aux_output')(gru_out_t)   

      # Body Only
      body_input = Input(name='body_input',shape=[max_len_b]) 
      body_Embed = Embedding(vocab_len_b+1,170,input_length=max_len_b,mask_zero=True,name='body_Embed')(body_input)
      gru_out_b = GRU(200)(body_Embed)

      # combined with GRU output
      com = concatenate([gru_out_t, gru_out_b])

      # now the combined data is being fed to dense layers
      dense1 = Dense(400,activation='relu')(com)
      dp1 = Dropout(0.5)(dense1)
      bn = BatchNormalization()(dp1) 
      dense2 = Dense(150,activation='relu')(bn)

      main_output = Dense(10, activation='sigmoid', name='main_output')(dense2)

      model = Model(inputs=[title_input, body_input],outputs=[main_output, auxiliary_output])
      return model

</br>

    model = RNN()
    model.summary()
    
    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    title_input (InputLayer)        [(None, 18)]         0                                            
    __________________________________________________________________________________________________
    body_input (InputLayer)         [(None, 600)]        0                                            
    __________________________________________________________________________________________________
    title_Embed (Embedding)         (None, 18, 2000)     71502000    title_input[0][0]                
    __________________________________________________________________________________________________
    body_Embed (Embedding)          (None, 600, 170)     81557500    body_input[0][0]                 
    __________________________________________________________________________________________________
    gru (GRU)                       (None, 300)          2071800     title_Embed[0][0]                
    __________________________________________________________________________________________________
    gru_1 (GRU)                     (None, 200)          223200      body_Embed[0][0]                 
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 500)          0           gru[0][0]                        
                                                                     gru_1[0][0]                      
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 400)          200400      concatenate[0][0]                
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 400)          0           dense[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 400)          1600        dropout[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 150)          60150       batch_normalization[0][0]        
    __________________________________________________________________________________________________
    main_output (Dense)             (None, 10)           1510        dense_1[0][0]                    
    __________________________________________________________________________________________________
    aux_output (Dense)              (None, 10)           3010        gru[0][0]                        
    ==================================================================================================
    Total params: 155,621,170
    Trainable params: 155,620,370
    Non-trainable params: 800

</br>

    model.compile(optimizer='adam',loss={'main_output': 'categorical_crossentropy', 'aux_output': 'categorical_crossentropy'},
              metrics=['accuracy'])
              
              
 </br>
 
    results=model.fit({'title_input': sequences_matrix_train_t, 'body_input': sequences_matrix_train_b},
          {'main_output': y_train, 'aux_output': y_train},
          validation_data=[{'title_input': sequences_matrix_test_t, 'body_input': sequences_matrix_test_b},
          {'main_output': y_test, 'aux_output': y_test}],
          epochs=3, batch_size=300)
          
 </br>
 
      (predicted_main, predicted_aux)=model.predict({'title_input': sequences_matrix_test_t, 'body_input': sequences_matrix_test_b},verbose=1)
      
  </br>
  
      from sklearn.metrics import classification_report,f1_score
      
      
      print(classification_report(y_test,predicted_main>.55))
      
     
         precision    recall  f1-score   support

               0       0.97      0.93      0.95      5936
               1       0.92      0.80      0.86      9087
               2       0.91      0.74      0.81      3863
               3       0.70      0.46      0.56      3219
               4       0.93      0.86      0.89      2564
               5       0.96      0.78      0.86      7818
               6       0.84      0.60      0.70      6905
               7       0.87      0.86      0.86      5990
               8       0.93      0.89      0.91      7433
               9       0.97      0.90      0.94      3495

       micro avg       0.91      0.79      0.85     56310
       macro avg       0.90      0.78      0.83     56310
    weighted avg       0.91      0.79      0.84     56310
     samples avg       0.84      0.82      0.82     56310
     
  </br>   
  
       test.iloc[24]
   </br>
  
    labels
    #Output
    array(['android', 'c#', 'c++', 'html', 'ios', 'java', 'javascript',
           'jquery', 'php', 'python'], dtype=object)
  
  </br>
  
      model.save('stackoverflow_tags.h5')

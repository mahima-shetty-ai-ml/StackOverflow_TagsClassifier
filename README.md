# StackOverflow_TagsClassifier

    import numpy as np # linear algebra
    import pandas as pd # data processing
    import collections
   
   </br>
   Importing necessary packages, numpy and pandas for performing a wide variety of mathematical operations on arrays and data manipulation and analysis.
   </br>
   
    ques = pd.read_csv('Questions.csv',encoding='iso-8859-1')
    ques.head(10)

   </br> Reading the Questions.csv from read.csv and obtaining first 10 rows of the same, to check the redundant columns so that we can remove them to improve our model.
   </br>
  
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
    So we found these columns which I find redundant and has to be removed.
    </br>
      * OwnerUserID : Any sort of categorical data which is very unique for every order or individual then it is to be removed because it produces no good contribution for model making.
      </br> 
      * Creation Date : Unless and until it's a time series analysis, there is no such importance or atleast in this analysis for our goal. So, this column is redundant for us.
      </br>
      * Closed Date :  Unless and until it's a time series analysis, there is no such importance or atleast in this analysis for our goal. So, this column is redundant for us.
      </br>
      * Score : We don't have any analysis to do with score, so hence this column is redundant for us.
      </br>

    ques.drop(["OwnerUserId","CreationDate","ClosedDate","Score"], axis=1, inplace=True)
    ques.head(10)
    
   </br> These columns can be removed with drop function. 
</br>
So we left with what we needed </br>

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

## Why am I using Regex here? 

</br>
As you can see specifically in column "Body", there are  "< p >" tags which is mandatory to remove or else our model will be get trained with it.
A RegEx, or Regular Expression, is a sequence of characters that forms a search pattern.
</br>
RegEx can be used to check if a string contains the specified search pattern.
</br>

    import re 

    def rem_html_tags(body):
        regex = re.compile('<.*?>')
        return re.sub(regex, '', body)
     
    ques['Body'] = ques['Body'].apply(rem_html_tags)
    ques.head()
  </br> Here we used .compile to find the pattern "< >"  and store it in regex variable and .sub function Replaces one or many matches with a space as we mentioned in the parameters.
    
 </br>
 Now we have Tags dataset in csv we can read it htrough read_csv() .head() shows the first 5 rows.
 </br>
 
     df_tags = pd.read_csv('Tags.csv', encoding='iso-8859-1')
     df_tags.head()
     
  </br>
  
## Importing important modules and packages for data manipulation and visualization.
  </br>
  
    import numpy as np # linear algebra
    import pandas as pd # data processing
    # Plotting Libs
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    %matplotlib inline

    import collections
    
 </br>
 Building color map and color array with respect to np.linspace (0, 0.9, num_plots)
 </br>
 
    #colormap = plt.cm.gist_ncar
    #fig = plt.figure().gca(projection='3d')
    #color_array = [colormap(i) for i in np.linspace(0, 0.9, num_plots)]
 
 </br>
 Building visualization and legend for the tags visualization 
 
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
    Converting tags column from dataset df_tags into a list and putting it in tag count counter and retrieving 10 most common tags out of it. Also plotting the tag counter info and check its data graphics. 
 </br>
 
    tagCount =  collections.Counter(list(df_tags['Tag'])).most_common(10)
    print(tagCount)
    plot_tags(tagCount)
    
 </br>
    
    
![image](https://user-images.githubusercontent.com/41589522/133224238-7e42dda6-d727-41b4-b2ea-6ae3553355be.png)

</br>
We get those top 10 common tags with us.
</br>

    top10=['javascript','java','c#','php','android','jquery','python','html','c++','ios']

</br>
    We have ID and top tags with us. We can merge these both datasets Tags and Questions on the basis of ID Column.
</br>

    tag_top10= df_tags[df_tags.Tag.isin(top10)]
    print (tag_top10.shape)
    tag_top10.head()

    total=pd.merge(ques, top10_tags, on='Id')
    print(total.shape)
    total.head()

</br>

## Understanding MultiLabel Binarizer 

### What is multi label Binarizer?
</br>
A set of labels (any orderable and hashable object) for each sample. If the classes parameter is set, y will not be iterated. Returns selfreturns this MultiLabelBinarizer instance fit_transform (y)[source] Fit the label sets binarizer and transform the given label sets.
</br> Data Credit: Scikit Learn Website </br>

![keras_multi_label_animation](https://user-images.githubusercontent.com/41589522/135709584-d179af3c-e9dc-4708-a04b-357376b43a96.gif)

</br>
Image Credit: PyImage Search
</br>

Now we import various packages and modules including NLP modules which we will use later.
</br>

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from nltk import word_tokenize
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing import sequence
    from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, BatchNormalization, GRU ,concatenate
    from keras.models import Model
</br>

From scikit-learn we will import multiLabelBinarizer for doing multilabel classification. .fit() -> fit() method, where we use the required formula and perform the calculation on the feature values of input data and fit this calculation to the transformer.


    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(total.Tags)
    labels = multilabel_binarizer.classes_
    labels  
    #output:
    array(['android', 'c#', 'c++', 'html', 'ios', 'java', 'javascript',
       'jquery', 'php', 'python'], dtype=object)
 
 </br>
 Splitting the data into train and test data using train_test_split
 
    train,test=train_test_split(total[:550000],test_size=0.25,random_state=24)
 
  </br>
  Segregating them into Title AND body, and applying .transform on train_tags for both train and test dependent variable. 
  
  
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

# Importing NLP based packages for doing language processing 

</br>

## Importing NLTK package

</br> NLTK, or Natural Language Toolkit, is a Python package that you can use for NLP. A lot of the data that you could be analyzing is unstructured data and contains human-readable text.

    import nltk
    nltk.download('punkt')


</br>

## What means by word tokenizer?

</br>
Word tokenization is the process of splitting a large sample of text into words. This is a requirement in natural language processing tasks where each word needs to be captured and subjected to further analysis like classifying and counting them for a particular sentiment etc.</br>

![image](https://user-images.githubusercontent.com/41589522/135713305-e1d88f43-6c8f-4b17-be9f-769c63b164ab.png)

</br>

    sent_lens_t=[]
    for sent in train['Title']:
        sent_lens_t.append(len(word_tokenize(sent)))
    max(sent_lens_t)  
    #output: 18  
 </br>
 
 Here we will tokenize the same with Tokenizer function. 
 
 
 ## What is text to sequences?
 
 
 Texts_to_sequences Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary.
</br>
Example </br>

   ![image](https://user-images.githubusercontent.com/41589522/135714065-86f4045e-91d2-4f77-bbba-0abbb0572671.png)

   ![image](https://user-images.githubusercontent.com/41589522/135714075-b71750f8-424e-40a7-8da9-98c9e0e16ca1.png)

 </br>Image credit: MLQ.ai
 
    max_len_t = 18
    tok = Tokenizer(char_level=False,split=' ')
    tok.fit_on_texts(X_train_t)
    sequences_train_t = tok.texts_to_sequences(X_train_t)
    
 </br>
 
 ## What is pad sequences?
 
 </br>
 Pad_sequences is used to ensure that all sequences in a list have the same length. By default this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence. </br>
 
 ![image](https://user-images.githubusercontent.com/41589522/135714556-b1d892e0-b246-43b2-b713-be3ab9a2c668.png)

  
    sequences_matrix_train_t = sequence.pad_sequences(sequences_train_t,maxlen=max_len_t)
    sequences_matrix_train_t
    
 </br>
 Same process with test dataset
 </br>
    sequences_test_t = tok.texts_to_sequences(X_test_t)
    sequences_matrix_test_t = sequence.pad_sequences(sequences_test_t,maxlen=max_len_t)
    
  </br>
  Taking sequences train and test data shape
  
  </br>
  
    sequences_matrix_train_t.shape,sequences_matrix_test_t.shape,y_train.shape,y_test.shape
    
  </br>
  Same process with the body of the training and testing dataset
  
  </br>
  
    sent_lens_b=[]
    for sent in train['Body']:
        sent_lens_b.append(len(word_tokenize(sent)))
    max(sent_lens_b)
    
    
    vocab_len_b =len(tok.index_word.keys())
    vocab_len_b

    
    sequences_matrix_train_b = sequence.pad_sequences(sequences_train_b,maxlen=max_len_b)
    sequences_matrix_train_b
  
    sequences_test_b = tok.texts_to_sequences(X_test_b)
    sequences_matrix_test_b = sequence.pad_sequences(sequences_test_b,maxlen=max_len_b)
    
    
  </br>
  
  # Model Training Part
  
  ## What is RNN?
  </br>
  Recurrent neural networks (RNN) are a class of neural networks that are helpful in modeling sequence data. Derived from feedforward networks, RNNs exhibit similar behavior to how human brains function. Simply put: recurrent neural networks produce predictive results in sequential data that other algorithms can't.
  
  
  ![image](https://user-images.githubusercontent.com/41589522/135717897-5a8c11eb-5b37-40ab-b79d-88be77953dea.png)
Image credit: Wikimedia Commons
</br>

## What is Embedding layer?

The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments:
</br>
It must specify 3 arguments:
</br>
input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.</br>
output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
</br>input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.


## Why GRU Layer?

To solve the vanishing gradient problem of a standard RNN, GRU is used. 

## Why Relu?

The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time.

## Dropout Layer

Dropout is a technique used to prevent a model from overfitting. Dropout works by randomly setting the outgoing edges of hidden units (neurons that make up hidden layers) to 0 at each update of the training phase.

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

## Adam Optimizer

Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.

Categorical crossentropy
Categorical crossentropy is a loss function that is used in multi-class classification tasks. These are tasks where an example can only belong to one out of many possible categories, and the model must decide which one. Formally, it is designed to quantify the difference between two probability distributions.

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

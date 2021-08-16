# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:34:07 2021

@author: Abhilash
"""

import tensorflow as tf
from tensorflow.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.keras.layers import Dropout, Input, Layer, Embedding, Reshape,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import networkx as nx
import scipy
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
import pandas as pd
class GraphConvolution(tf.keras.layers.Layer):  # ReLU(AXW)

    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, 
                 seed=1024, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.seed = seed
       
        
        
    def build(self, input_shapes):
        input_dim = int(input_shapes[0][-1])
        
        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      name='kernel' )
        
        self.bias = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        name='bias')

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.built = True
        print(f"kernel shape {self.kernel.shape}")
        print(f"input dimension {input_dim}")

    def call(self, inputs, training=None, **kwargs):        
        features, A = inputs
        A=tf.sparse.to_dense(A)
        output = tf.matmul(A,self.kernel)
        output += self.bias
        act = self.activation(output)
        return act

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'seed': self.seed
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def GCN(adj_dim,feature_dim,n_hidden, num_class, num_layers,activation=tf.nn.relu,dropout_rate=0.5, l2_reg=0 ):
    Adj = Input(shape=(None,), sparse=True,name='first')
    
    X_in = Input(shape=(feature_dim,), sparse=False,name='second')
    emb = Embedding(adj_dim, feature_dim,
                        embeddings_initializer=Identity(1.0), trainable=False)
    X_emb = emb(X_in)
    H=X_emb
    for i in range(num_layers):
        if i == num_layers - 1:
            activation = tf.nn.softmax
            n_hidden = num_class
        h = GraphConvolution(n_hidden, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg)([H,Adj])
    output = h
    model = Model(inputs=[X_in,Adj], outputs=output)
    #logging.info(model.summary())
    return model

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = scipy.sparse.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = scipy.sparse.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + scipy.sparse.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def get_gcn_embeddings(hidden_units,train_df_temp,source_label,target_label,epochs,num_layers,subset):
    label_set=[]
    if(subset<train_df_temp.index.size):
        train_df=train_df_temp[:subset]
        graph=nx.from_pandas_edgelist(train_df,source=source_label,target=target_label)
        if(graph.number_of_nodes()>subset ):
            label_set=train_df_temp[target_label][:graph.number_of_nodes()].tolist()
        else:
            label_set=train_df[target_label][:graph.number_of_nodes()].tolist()
    else:
        graph=nx.from_pandas_edgelist(train_df_temp[:],source=source_label,target=target_label)
        if(graph.number_of_nodes()>subset ):
            temp_list=train_df_temp[target_label][:].tolist()
            for i in range(graph.number_of_nodes()-subset):
                label_set.append(temp_list[-1])
        else:
            label_set=train_df_temp[target_label][:graph.number_of_nodes()].tolist()

    
    print(f"Graph with {graph.number_of_nodes()} nodes should have the same labels")
    A=nx.adjacency_matrix(graph,nodelist=range(graph.number_of_nodes()))
    A=preprocess_adj(A)
    label_y= LabelEncoder()
    
    labels=label_y.fit_transform(label_set)
    y_train=encode_onehot(labels)
    print(f"Created Laplacian {A}")
    feature_dim = A.shape[-1]
    X = np.arange(A.shape[-1])
    X_n=[]
    for i in range(feature_dim):
        X_n.append(X)
    X=np.asarray(X_n)
    model_input = [X, A]

    print(f"feature_dim {feature_dim}")
    
    model = GCN(A.shape[-1],feature_dim, hidden_units, y_train.shape[-1],num_layers,  dropout_rate=0.5, l2_reg=2.5e-4 )
    model.compile(optimizer='adam', loss='categorical_crossentropy',weighted_metrics=['categorical_crossentropy', 'acc'])
    print(model.summary())
    print("Fitting model with {hidden_units} units")
    model.fit(model_input,y_train[:A.shape[-1]],epochs=epochs)
    
    embedding_weights = model.predict(model_input)
    print(f"Dimensions of embeddings {embedding_weights.shape}")
    
    print(embedding_weights)
    return embedding_weights
def get_node_embedding(node,embedding_weights):
    try:
        print(f"Dimension of embedding is denoted by the number of labels {embedding_weights.shape[1]}")
        return embedding_weights[node]
    except:
        print(f"Value of node should be in between 0 and {embedding_weights.shape[0]}")


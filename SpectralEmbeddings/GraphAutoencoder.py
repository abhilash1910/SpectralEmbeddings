# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 12:56:12 2021

@author: Abhilash
"""

import tensorflow as tf
import networkx as nx
import numpy as np
import scipy
import logging

class Loss():
    def reconstruction_loss(self,beta):
        self.beta=beta
        def cal(y_pred,y_true):
            delta=tf.square((y_pred-y_true)*self.beta)
            return tf.reduce_sum(delta)
        return cal
    def loss_laplace(self,alpha):
        self.alpha=alpha
        def cal(y_true, y_pred):
            L = y_true
            Y = y_pred
            batch_size = tf.cast(tf.keras.backend.shape(L)[0], tf.float32)
            return self.alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size
        return cal

class SDNE():
    def __init__(self,graph,alpha,beta,hidden_dims,epochs):
        self.graph=graph
        self.alpha=alpha
        self.beta=beta
        self.hidden_dims=hidden_dims
        self.A=nx.adjacency_matrix(self.graph,nodelist=range(self.graph.number_of_nodes()))
        idx=np.arange(self.graph.number_of_nodes())
        degree_vals=np.array([(self.graph.degree[node]) for node in list(self.graph.nodes())])
        n=self.graph.number_of_nodes()
        self.D=scipy.sparse.coo_matrix((degree_vals,(idx,idx)),shape=(n,n))
        
        L=self.D-self.A
        L=nx.from_scipy_sparse_matrix(L)
        self.L=nx.laplacian_matrix(L)
        self.inputs=[self.A,self.L]
        self.embeddings=[]
        self.node_size=self.graph.number_of_nodes()
        self.epochs=epochs
        
        
    def create_model(self): 
        node_size=self.graph.number_of_nodes()
        A = tf.keras.layers.Input(shape=(node_size,))
        L = tf.keras.layers.Input(shape=(None,))
        encoder_module = A
        print(f"Encoder Layer with {len(self.hidden_dims)} layers")
        for i in range(len(self.hidden_dims)):
            if i == len(self.hidden_dims) - 1:
                encoder_module = tf.keras.layers.Dense(self.hidden_dims[i], activation='sigmoid', name='encoders')(encoder_module)
            else:
                encoder_module = tf.keras.layers.Dense(self.hidden_dims[i], activation='relu')(encoder_module)
        print(f"Decoder Layer with {len(self.hidden_dims)} layers")
        Y = encoder_module
        for i in reversed(range(len(self.hidden_dims) - 1)):
            decoder_module = tf.keras.layers.Dense(self.hidden_dims[i], activation='relu')(encoder_module)

        A_ = tf.keras.layers.Dense(node_size, 'relu', name='decoder')(decoder_module)
        self.models = tf.keras.Model(inputs=[A, L], outputs=[A_, Y])
        self.emb = tf.keras.Model(inputs=A, outputs=Y)
        return self.models, self.emb

    def model(self, opt='adam', initial_epoch=0, verbose=1):
        self.models, self.emb_model = self.create_model()
        loss=Loss()
        self.models.compile(opt, [loss.reconstruction_loss(self.beta), loss.loss_laplace(self.alpha)])
        self.get_embeddings()
        batch_size = self.node_size
        print(f"AutoEncoder Model with {[self.A.todense(), self.L.todense()]} inputs")
        return self.models.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()],
                                  batch_size=batch_size, epochs=self.epochs, initial_epoch=initial_epoch, verbose=verbose)
    
    def get_embeddings(self):
        logging.info("Creating node embedding matrix with SDNE autoencoder")
        embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
        for _,emb in enumerate(embeddings):
            self.embeddings.append(emb)
        return self.embeddings   
    
    def node_level_embedding(self,node,embed):
        logging.info("Determining the Chebyshev distance between node and rest of the node embeddings")
        embed_node=embed[node]
        vals=list(self.graph.nodes())
        def chebyshev_distance(node1,node2):
            return scipy.spatial.distance.chebyshev(node1,node2)
        distances=[]
        questions=[]
        for i in range(self.graph.number_of_nodes()):
            if i!=node:
                distances.append(chebyshev_distance(embed_node,embed[i]))
                questions.append(vals[i])
        return vals[node],distances,questions

def get_sdne_embeddings(train_df,source_label,target_label,hidden_dims,alpha,beta,epochs):
    g=nx.from_pandas_edgelist(train_df,source=source_label,target=target_label)
    print(f"Graph with {g.number_of_nodes()} nodes should have the same labels")
    #hidden_dims=[32,16]
    #alpha=1e-4
    #beta=1e-5
    sdne=SDNE(g,alpha,beta,hidden_dims,epochs)
    sdne.model()
    emb=sdne.get_embeddings()

    #node_num=339
    #node,distances,questions=sdne.node_level_embedding(node_num,emb)
    #sdne_df=pd.DataFrame(columns=['Question','Sample_Question','Chebyshev_Distance'])
    #sdne_df['Question']=[node]*len(distances)
    #sdne_df['Sample_Question']=questions
    #sdne_df['Chebyshev_Distance']=distances
    return emb
    

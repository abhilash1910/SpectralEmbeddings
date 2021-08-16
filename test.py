# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:44:16 2021

@author: Abhilash
"""

import pandas as pd
import numpy as np
import SpectralEmbeddings.VanillaGCN as vgcn
import SpectralEmbeddings.ChebGCN as cgcn
import SpectralEmbeddings.SplineGCN as sgcn
import SpectralEmbeddings.GraphAutoencoder as graph_ae

def test_vanillagcn():
    print("Testing for VanillaGCN embeddings having a source and target label")
    train_df=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden units , number of layers,subset (values of entries to be considered for embeddings),epochs ")
    hidden_units=32
    num_layers=4
    subset=34
    epochs=10
    v_emb=vgcn.get_gcn_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,subset)
    print(v_emb.shape)
    
def test_chebgcn():
    print("Testing for ChebGCN embeddings having a source and target label")
    train_df=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden units , number of layers,subset (values of entries to be considered for embeddings),epochs and k for Cheby polynomials")
    hidden_units=32
    num_layers=4
    subset=34
    epochs=10
    k=4
    c_emb=cgcn.get_chebgcn_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,subset,k)
    print(c_emb.shape)
    
def test_sgcn():
    print("Testing for SplineGCN embeddings having a source and target label")
    train_df=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden units , number of layers,subset (values of entries to be considered for embeddings),epochs and k for Cheby polynomials")
    hidden_units=32
    num_layers=4
    subset=34
    epochs=10
    s_emb=sgcn.get_splinegcn_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,subset)
    print(s_emb.shape)
    
def test_graph_ae():
    print("Testing for Graph Autoencoder embeddings having a source and target label")    
    train_df=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden dimensions ,alpha,beta,epochs")   
    hidden_dims=[32,16]
    alpha=1e-4
    beta=1e-5
    epochs=20
    g_emb=graph_ae.get_sdne_embeddings(train_df,source_label,target_label,hidden_dims,alpha,beta,epochs)
    print(g_emb)

if __name__=='__main__':
    """Embeddings generated from GCN variants are of the dimensions of (input subset size,set of labels)-> in this case if input
       subset size is 20 and number of labels are 5 then the embedding dimension -> (20,5)
       
       Embedding generated from Graph Autoencoder are of the dimension (input subset size,dimension of autoencoder hidden units)
       """
    print("======Vanilla GCN========")    
    test_vanillagcn()
    print("======Chebshev GCN========")
    test_chebgcn()
    print("======Spline GCN========")
    test_sgcn()
    print("======Graph AutoEncoder========")
    test_graph_ae()
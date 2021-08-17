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
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot,plot
import plotly
import plotly.graph_objs as go
import networkx as nx
from pyvis.network import Network
init_notebook_mode(connected=True)


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
    v_emb,v_graph=vgcn.get_gcn_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,subset)
    print(v_emb.shape)
    return v_emb,v_graph
    
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
    c_emb,c_graph=cgcn.get_chebgcn_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,subset,k)
    print(c_emb.shape)
    return c_emb,c_graph
    
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
    s_emb,s_graph=sgcn.get_splinegcn_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,subset)
    print(s_emb.shape)
    return s_emb,s_graph
    
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
    g_emb,graph_ae_pl=graph_ae.get_sdne_embeddings(train_df,source_label,target_label,hidden_dims,alpha,beta,epochs)
    print(g_emb)
    return g_emb,graph_ae_pl

def plotter(G,title):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='white'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='ice',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=title,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                plot_bgcolor='#222222',
                annotations=[ dict(
                    text="No. of connections",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plot(fig)    
    
def plot_vgcn_embed(graph,node_num,emb,label):
    
    node,distances,questions=vgcn.node_level_embedding(graph,node_num,emb)
    vg_df=pd.DataFrame()
    vg_df['Premise']=[node]*len(distances)
    vg_df['Hypothesis']=questions
    vg_df['Chebyshev_Distance']=distances
    vg_g=nx.from_pandas_edgelist(vg_df,source='Hypothesis',target='Premise',edge_attr='Chebyshev_Distance')
    plotter(vg_g,label)
    return vg_g

def plot_cgcn_embed(graph,node_num,emb,label):
    
    node,distances,questions=cgcn.node_level_embedding(graph,node_num,emb)
    vg_df=pd.DataFrame()
    vg_df['Premise']=[node]*len(distances)
    vg_df['Hypothesis']=questions
    vg_df['Chebyshev_Distance']=distances
    vg_g=nx.from_pandas_edgelist(vg_df,source='Hypothesis',target='Premise',edge_attr='Chebyshev_Distance')
    plotter(vg_g,label)
    return vg_g


def plot_sgcn_embed(graph,node_num,emb,label):
    
    node,distances,questions=sgcn.node_level_embedding(graph,node_num,emb)
    vg_df=pd.DataFrame()
    vg_df['Premise']=[node]*len(distances)
    vg_df['Hypothesis']=questions
    vg_df['Chebyshev_Distance']=distances
    vg_g=nx.from_pandas_edgelist(vg_df,source='Hypothesis',target='Premise',edge_attr='Chebyshev_Distance')
    plotter(vg_g,label)
    return vg_g

def plot_ae_embed(graph,node_num,emb,label):
    
    node,distances,questions=graph_ae.node_level_embedding(graph,node_num,emb)
    vg_df=pd.DataFrame()
    vg_df['Premise']=[node]*len(distances)
    vg_df['Hypothesis']=questions
    vg_df['Chebyshev_Distance']=distances
    vg_g=nx.from_pandas_edgelist(vg_df,source='Hypothesis',target='Premise',edge_attr='Chebyshev_Distance')
    plotter(vg_g,label)
    return vg_g

def pyvis_plotter(graph,label):
    network = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    network.from_nx(graph)
    #network.enable_physics(True)
    #network.show_buttons(filter_=['nodes'])
    network.show('label.html')
    
if __name__=='__main__':
    """Embeddings generated from GCN variants are of the dimensions of (input subset size,set of labels)-> in this case if input
       subset size is 20 and number of labels are 5 then the embedding dimension -> (20,5)
       
       Embedding generated from Graph Autoencoder are of the dimension (input subset size,dimension of autoencoder hidden units)
    
       """
    
    print("======Vanilla GCN========")    
    
    embed_wt,v_graph=test_vanillagcn()
    node_num=12 #node number for plotting
    if node_num>v_graph.number_of_nodes():
        print('The node number should not be greater than number of nodes in graph')
        node_num=v_graph.number_of_node()-1  
    label="Vanilla GCN Chebshev similarity"
    v_g=plot_vgcn_embed(v_graph,node_num,embed_wt,label)
    #pyvis_plotter(v_graph,'VanillaGCN')
    
    print("======Chebshev GCN========")
    
    embed_wt_cheb,c_graph=test_chebgcn()
    node_num=12 #node number for plotting
    if node_num>c_graph.number_of_nodes():
        print('The node number should not be greater than number of nodes in graph')
        node_num=c_graph.number_of_node()-1 
    label="Chebyshev GCN Chebshev similarity"
    plot_cgcn_embed(c_graph,node_num,embed_wt_cheb,label)
    
    print("======Spline GCN========")
    
    embed_wt_spline,s_graph=test_sgcn()
    node_num=12 #node number for plotting
    if node_num>s_graph.number_of_nodes():
        print('The node number should not be greater than number of nodes in graph')
        node_num=s_graph.number_of_node()-1    
    label="Spline GCN Chebshev similarity"
    c_g=plot_sgcn_embed(s_graph,node_num,embed_wt_spline,label)
    pyvis_plotter(s_graph,'SplineGCN')
    
    
    print("======Graph AutoEncoder========")
    
    graph_ae_embed,ae_graph=test_graph_ae()
    node_num=12 #node number for plotting
    if node_num>ae_graph.number_of_nodes():
        print('The node number should not be greater than number of nodes in graph')
        node_num=ae_graph.number_of_node()-1    
    label="Graph Autoencoder Chebschev similarity"
    plot_ae_embed(ae_graph,node_num,graph_ae_embed,label)
    #pyvis_plotter(temp_g,'Graph_AE')
    
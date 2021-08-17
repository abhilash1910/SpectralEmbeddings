# SpectralEmbeddings


## A Spectral Embedding library made of Graph Convolution Networks and AutoEncoders :robot:


This is a embedding generator library used for creating Graph Convolution Network, and Graoh Autoencoder embeddings from Knowledge Graphs. This allows projection of higher order network dependencies for creating the node embeddings with respect to a neighborhood. There are 2 different approaches: 

- Graph AutoEncoder Approach: This models the first and higher order similarity measures in a graph for each node in a neighborhood. The first and second order similarity measures are created through an Autoencoder circuit which preserves the proximity loss of similarity with reconstruction loss.

<img src="https://www.programmersought.com/images/979/223a8a8bc9b82f9255018d248c355c8b.png">
 
 ![img1](Previews/Graph_AE_preview.PNG)


- Graph Convolution Network Variants: These include VanillaGCN,ChebGCN and SplineGCN kernels which provide spectral embeddings from a knowledge graph.


 - VanillaGCN: The steps to produce this include ,creating the adjacency matrix representation along with the node features from the inputs. The labels have to be one hot encoded to maintain the dimensions of the inputs. The model inputs are in the form of [node features,adjacency matrix] representation and the outputs are [one hot encoded node labels]. This matrix is then processed and additional layers such as Embedding Layer/LSTM can be added to perform node classification. We extract the penultimate layer for getting the embeddings in this case.

   <img src="https://miro.medium.com/max/875/1*THVRB8-wHODA3yDUykasIg.png">

   ![img2](Previews/Vanilla_GCN_preview_1.PNG)

  - SplineGCN: Spline GCN involve computing smooth spectral filters to get localized spatial filters. The connection between smoothness in frequency domain and localization in space is based on Parseval’s Identity (also Heisenberg uncertainty principle): smaller derivative of spectral filter (smoother function) ~ smaller variance of spatial filter (localization) In this case, we wrap the vanilla GCN with an additional spline functionality by decomposing the laplacian to its diagonals (1-spline) . This represents the eigenvectors which can be added independently instead of taking the entire laplacian at one time. The rest of the code segment remains the same.

  <img src="https://miro.medium.com/max/1838/1*--D1tDMjYWwf1mv8ZYRo7A.png">
  
  ![img3](Previews/Spline_GCN_preview.PNG)


  - ChebGCN: This is one of the most important parts of spectral GCN where Chebyshev polynomials are used instead of the laplacian. ChebNets are GCNs that can be used for any arbitrary graph domain, but the limitation is that they are isotropic. Standard ConvNets produce anisotropic filters because Euclidean grids have direction, while Spectral GCNs compute isotropic filters since graphs have no notion of direction (up, down, left, right).

  <img src="https://atcold.github.io/pytorch-Deep-Learning/images/week13/13-2/Figure2.png">

  ![img4](Previews/Chebyshev_GCN_preview.PNG)





## Dependencies

<a href="https://www.tensorflow.org/">Tensorflow</a>


<a href="https://networkx.org/">Networkx</a>


<a href="https://scipy.org/">scipy</a>


<a href="https://scikit-learn.org/stable/">sklearn</a>



## Usability

Installation is carried out using the pip command as follows:

```python
pip install SpectralEmbeddings==0.1
```


## Samples


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT

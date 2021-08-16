# SpectralEmbeddings


## A Spectral Embedding library made of Graph Convolution Networks and AutoEncoders :robot:


This is a topic  clustering library built with transformer eembeddings and analysing cosine similarity between them. The topics are clustered either by kmeans or agglomeratively depending on the use case, and the embeddings are attained after propagating through any of the Transformers present in [HuggingFace](https://huggingface.co/transformers/pretrained_models.html).The library can be found [here](https://pypi.org/project/ClusterTransformer/).



## Dependencies

<a href="https://pytorch.org/">Pytorch</a>


<a href="https://huggingface.co/transformers/">Transformers</a>





## Usability

Installation is carried out using the pip command as follows:

```python
pip install ClusterTransformer==0.1
```

For using inside the Jupyter Notebook or Python IDE:

```python
import ClusterTransformer.ClusterTransformer as ct
```

The  'ClusterTransformer_test.py' file contains an example of using the Library in this context.

## Samples

[Colab-Demo](https://colab.research.google.com/drive/18HAoATFfuXGAGzPcOhWgZa0a9B6yOpKK?usp=sharing)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT

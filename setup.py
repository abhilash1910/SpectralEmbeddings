# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:20:30 2021

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'SpectralEmbeddings',         
  packages = ['SpectralEmbeddings'],   
  version = '0.2',       
  license='MIT',        
  description = 'A Spectral Embedding library made of Graph Convolution Networks and AutoEncoders',   
  long_description='This is a library used for generating semantic embeddings from Graph Convolution variants such as Chebscev polynomials and AutoEncoders. The embeddings generated are based on deep geaph networks which use spectral convolution kernel made with Tensorflow. This package implements AutoEncoder based embeddings as well as GCN variants',
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/SpectralEmbeddings',   
  download_url = 'https://github.com/abhilash1910/SpectralEmbeddings/archive/v_02.tar.gz',    
  keywords = ['Semantic Embeddings','Graph Convolution Network','Chebyshev networks','Higher order Graph embeddings','Graph Autoencoder networks','SDNE embeddings','Tensorflow'],   
  install_requires=[           

          'numpy',         
          'tensorflow',
          'keras',
          'sklearn',
          'pandas',
          'networkx',
          'scipy',
          'plotly'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',      
    'Programming Language :: Python :: 3.8',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

# protein-secondary-structure-prediction-with-CNN-and-GCN
This project is to predict the protein secondary structure for 8-state prediction (8 state) using Deep Learning - Convolutional Neural Networks and Graph Convolutional Networks.

Protein structure prediction is one of the crucial issues in present day computational biology. The primary structure of a protein is its amino acid sequence and the secondary structure of a protein can be used to predict the tertiary structure. Alpha (α) helix and beta (β) sheet are the most common secondary protein structures. The secondary structure prediction is a set of techniques in bioinformatics that aims to predict the local secondary structure of protein. There are many emerging methods (supervised or unsupervised machine learning) to address the problem of protein secondary structure prediction (PSSP). In this paper, we will review the use of deep learning neural networks, as they are powerful methods for studying such large data sets and has shown superior performance in many areas of machine learning. We used the development and application on deep learning neural networks, which are CNN and GCN, to predict the secondary structure of protein using the amino acid sequences as inputs. Our results confirm that the presence of amino acids in the protein sequence increases the stability for the approximation of the secondary structure of the protein.

Convolutional neural networks are current state-of-art architecture for image or text classification tasks. CNN is being used everywhere, be it for processing sequential data such as audio, time series, NLP and in this session we use this algorithm to predict the secondary structure. The term convolution on CNN refers to the mathematical combination of two functions to produce a third function and then it combines two sets of information. There are 3 types of convolution operations. 1D convolution (used where input is sequential such as text or audio); 2D convolution (used where the input is an image) and 3D convolution (used in 3D medical imaging or detecting events in video). CNN helps extract features from text/images that can assist in data processing for prediction, by extracting low-dimensional features, and then some high-dimensional features such as shapes.

<img width="345" alt="image" src="https://user-images.githubusercontent.com/71427418/169717813-0d3c5110-8170-4755-bc83-49574cff50b4.png">
An overview of convolutional neural network of our model.


Graph Convolutional Networks
Convolutional neural networks can solve problems with ordinary 1-D and 2-D Euclidean data such as image and text classification, but often real-world data has a non-Euclidean structure. In this stage, graph neural networks become a solution that allows us to capture rich features of complex relationships between data. In recent years, various variants of graph neural networks are being developed with graph convolutional networks (GCN) being one of them. 

<img width="274" alt="image" src="https://user-images.githubusercontent.com/71427418/169717826-1c48fa16-5840-41af-8de3-702f44ccb8a1.png">
Illustration of graph convolutional networks (GCN).


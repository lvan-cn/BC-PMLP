Transfer Learning from Rating Prediction to Top-k Recommendation
=======
This repository contains the codes and data necessary to reproduce the results of the following article:

Fan Ye, Xiaobo Lu, Hongwei Li, Zhenyu Chen, "Transfer Learning from Rating Prediction to Top-k Recommendation," submitted.

<h1>Abstract</h1>

Recommender system has made great strides in two major research fields, rating prediction and Top-k recommendation. In essence, rating prediction is a regression task, which aims to predict users scores on other items, while Top-k is a classification task selecting the items that users have the most potential to interact with. Both characterize users and items, but the optimization of parameters varies widely for their respective tasks. Inspired by the idea of transfer learning, we consider extracting the information learned from rating prediction models for serving for Top-k tasks. To this end, we propose a universal transfer model for recommender systems. The transfer model consists of two sub-components: quadruple-based Bayesian Converter (BC) and Prediction-based Multi-Layer Perceptron (PMLP). As the main part, BC is responsible for transforming the feature vectors extracted from the rating prediction model. Meanwhile, PMLP extracts the prediction ratings, constructs the prediction rating matrix, and uses multi-layer perceptron to enhance the final performance. On four benchmark datasets, we use the information extracted from the singular value decomposition plus plus (SVD++) model to demonstrate the effectiveness of BC-PMLP, comparing to classical and state-of-the-art baselines. We also conduct extra experiments to verify the utility of BC, and performance within different parameter values.

<h1>Problem formulation</h1>

In this section, we will focus on the preliminary  definition of transfer learning from rating prediction problem to Top-k task. The question of rating prediction is how you predict unknown user ratings from known user history. In Top-k recommendation, K items are recommended to the user, and these recommendations are presented to the user in descending order based on the user's "rating" of the item. For example, when you browse Amazon, the site will recommend K items that you are most likely to buy. Transfer learning is a machine learning method that transfers the knowledge learned through Ts tasks in the source domain to Tt tasks in the target domain to improve the performance of Tt task model prediction. The task of transfer learning is to start from the similarity, find the similarity of the target problem, and apply the model learned in the old domain to the new domain.

Overall, transfer learning for recommender system faces two main challenges:

The first challenge is the definition of transfer learning in recommender system. This issue mainly includes what information should be transferred and how the information is processed. The solutions determine the transfer algorithm. 

The second challenge is to find out whether both explicit ratings and implicit interactions play an important role in the transfer process, and how to perform different transfer treatments and combine the two.

## Repository structure

The pository consists of the following folders:

    - Folder scripts: 
    	Code BC_PMLP.py: Store the construction code of PMLP (called REG) and the prediction code of BC (called CLA);
    	Code Dataset.py: Some data preprocessing functions. If you run this code directly, this function file will not work;
    	Code run.py: There is a main function inside.
    	Code SVD++.py: The source code of SVD++ algorithm. After running this program, an embedding vector file will be generated.
    	
    - Folder data: 
    	Folder temp: Store the feature vectors trained by BC and derived from SVD++ in the form of dictionary; This will help you verify the effectiveness of strategies.
    	Four test data and Four training data are stored in it. We do not provide the original data in the whole folder, but only the encoded and segmented processed data.

More details about the code are given in the form of comments in the code file.

<h1>Dataset</h1>

The data sets required for the experiment are all in the data folder.

The original data set is obtained as follows: 

The Movielens-1M dataset is available at https://grouplens.org/datasets/movielens/1m/. 

The Netflix dataset is available at https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data.

The Yelp dataset is available at https://www.yelp.com/dataset. 

The FilmTrust dataset is available at https://guoguibing.github.io/librec/datasets.html.

Environment Requirement
---------------------

The general environment of the experiment is given as follows： 

- Python 3.5
- torch 1.4.0
- numpy 1.19.2
- scipy 1.1.0
- tqdm 4.64.1

Example to Run the Codes
-------------------------

Before transfer learning, we need to get the embedding vectors trained by SVD++. You can execute the following code:
```
python SVD++.py
```
Later, the program will generate a file called `SVD++_emb_file_Netflix` based on the training data. Then, you can successfully get a set of experimental results with the following instructions：
```
python run.py
```
The parameter values required for program operation can be modified in the source code.

​	**user_num** is the number of users.

​	**item_num** is the number of items.

​	**file** or **data** is the name of the data set.

​	**steps** is the number of steps in SVD ++ training.

​	**mlp_epoch** is the training epoch of PMLP.

​	**mlp_layers** is the dimensions of each network layer.

​	**alpha** is the balance factor.

<h1>License</h1>

This code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree. If you are using this code in any way for research that results in a publication, please cite the article above.

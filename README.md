# RecommenderSystems
## Collaborative Filtering
### Introduction
The increasing importance of the web as a medium for electronic and business transactions has served as a driving force for the development of recommender systems technology. An important catalyst in this regard is the ease with which the web enables users to provide feedback about their likes or dislikes. The basic idea of recommender systems is to utilize these user data to infer customer interests. The entity to which the recommendation is provided is referred to as the user, and the product being recommended is referred to as an item.  
The basic models for recommender systems works with two kinds of data:    
1. User-Item interactions such as ratings  
2. Attribute information about the users and items such as textual proﬁles or relevant keywords  
Models that use type 1 data are referred to as collaborative ﬁltering methods, whereas models that use type 2 data are referred to as content based methods. In this project, we will build recommendation system using collaborative ﬁltering methods. 

### Collaborative Filtering Models 

Collaborative ﬁltering models use the collaborative power of the ratings provided by multiple users to make recommendations. The main challenge in designing collaborative ﬁltering methods is that the underlying ratings matrices are sparse. Consider an example of a movie application in which users specify ratings indicating their like or dislike of speciﬁc movies. Most users would have viewed only a small fraction of the large universe of available movies and as a result most of the ratings are unspeciﬁed.  
The basic idea of collaborative ﬁltering methods is that these unspeciﬁed ratings can be imputed because the observed ratings are often highly correlated across various users and items. For example, consider two users named John and Molly, who have very similar tastes. If the ratings, which both have speciﬁed, are very similar, then their similarity can be identiﬁed by the ﬁltering algorithm. In such cases, it is very likely that the ratings in which only one of them has speciﬁed a value, are also likely to be similar. This similarity can be used to make inferences about incompletely speciﬁed values. Most of the collaborative ﬁltering methods focuses on leveraging either inter-item correlations or inter-user correlations for the prediction process.  

In this project, we will implement and analyze the performance of two types of collaborative ﬁltering methods:  
1. Neighborhood-based collaborative ﬁltering  
2. Model-based collaborative ﬁltering  

### Movie Lens Dataset

In this project, we will build a recommendation system to predict the ratings of the movies in the MovieLens dataset. The dataset can be downloaded using the following link:  
http://files.grouplens.org/datasets/movielens/ml-latest-small.zip  
Although the dataset contains movie genre information, but we will only use the movie rating information in this project. For the subsequent discussion, we assume that the ratings matrix is denoted by R, and it is an m×n matrix containing m users (rows) and n movies (columns). The (i,j) entry of the matrix is the rating of user i for movie j and is denoted by rij. Before moving on to the collaborative ﬁlter implementation, we will analyze and visualize some properties of this dataset  

#### Task 1 - Compute the sparsity of the movie rating dataset, where sparsity is deﬁned by equation 1  
    Sparsity = Total number of available ratings/ Total number of possible ratings

#### Task 2 -  Plot a histogram showing the frequency of the rating values. To be speciﬁc, bin the rating values into intervals of width 0.5 and use the binned rating values as the horizontal axis. Count the number of entries in the ratings matrix R with rating values in the binned intervals and use this count as the vertical axis. Brieﬂy comment on the shape of the histogram

#### Task 3 - Plot the distribution of ratings among movies. To be speciﬁc, the X-axis should be the movie index ordered by decreasing frequency and the Y -axis should be the number of ratings the movie has received.  

#### Task 4 - Plot the distribution of ratings among users. To be speciﬁc, the X-axis should be the user index ordered by decreasing frequency and the Y -axis should be the number of movies the user have rated.  

#### Task 5 - Explain the salient features of the distribution found in question 3 and their implications for the recommendation process.

#### Task 6 -  Compute the variance of the rating values received by each movie. Then, bin the variance values into intervals of width 0.5 and use the binned variance values as the horizontal axis. Count the number of movies with variance values in the binned intervals and use this count as the vertical axis. Brieﬂy comment on the shape of the histogram  

### Neighborhood-based collaborative ﬁltering

The basic idea in neighborhood-based methods is to use either user-user similarity or item-item similarity to make predictions from a ratings matrix. There are two basic principles used in neighborhood-based models:  
1. User-based models: Similar users have similar ratings on the same item. Therefore, if John and Molly have rated movies in a similar way in the past, then one can use John’s observed ratings on the movie Terminator to predict Molly’s rating on this movie.  
2. Item-based models: Similar items are rated in a similar way by the same user. Therefore, John’s ratings on similar science ﬁction movies like Alien and Predator can be used to predict his rating on Terminator.  
In this project, we will only implement user-based collaborative ﬁltering (implementation of item-based collaborative ﬁltering is very similar).  

#### User-based neighborhood models
In this approach, user-based neighborhoods are deﬁned in order to identify similar users to the target user for whom the rating predictions are being computed. In order to determine the neighborhood of the target user u, her similarity to all the other users is computed. Therefore, a similarity function needs to be deﬁned between the ratings speciﬁed by users. In this project, we will use Pearson-correlation coeﬃcient to compute the similarity between users  

#### Pearson-correlation coeﬃcient

Pearson-correlation coeﬃcient between users u and v, denoted by Pearson(u,v), captures the similarity between the rating vectors of users u and v. Before stating the formula for computing Pearson(u,v), let’s ﬁrst introduce some notation:  

Iu : Set of item indices for which ratings have been speciﬁed by user u   
Iv : Set of item indices for which ratings have been speciﬁed by user v   
µu:  Mean rating for user u computed using her speciﬁed ratings   
ruk: Rating of user u for item k

#### Task 7 - Write down the formula for µu in terms of Iu and ruk
#### Task 8 -  In plain words, explain the meaning of Iu ∩Iv. Can Iu ∩Iv = ∅ (Hint: Rating matrix R is sparse).

#### k-Nearest neighborhood (k-NN)  
Having deﬁned similarity metric between users, now we are ready to deﬁne neighborhood of users. k-Nearest neighbor of user u, denoted by Pu, is the set of k users with the highest Pearson-correlation coeﬃcient with user u

### Prediction Function 
We can now deﬁne the prediction function for user-based neighborhood model. The predicted rating of user u for item j, denoted by ˆ ruj, is given by equation 3

#### Task 9 - Can you explain the reason behind mean-centering the raw ratings (rvj − µv) in the prediction function? (Hint: Consider users who either rate all items highly or rate all items poorly and the impact of these users on the prediction function)

#### k-NN collaborative filter
The previous sections have equipped you with the basics needed to implement a k-NN collaborative ﬁlter for predicting ratings of the movies. Although, we have provided you with the equations needed to write a function for predicting the ratings but we don’t require you to write it. Instead, you can use the built-in python functions for prediction.

#### Design and test via cross-validation
In this part of the project, you will design a k-NN collaborative ﬁlter and test it’s performance via 10-fold cross validation. In a 10-fold cross-validation, the dataset is partitioned into 10 equal sized subsets. Of the 10 subsets, a single subset is retained as the validation data for testing the ﬁlter, and the remaining 9 subsets are used to train the ﬁlter. The cross-validation process is then repeated 10 times, with each of the 10-subsets used exactly once as the validation data.  

#### Task 10 - Design a k-NN collaborative ﬁlter to predict the ratings of the movies in the MovieLens dataset and evaluate it’s performance using 10-fold cross validation. Sweep k ( number of neighbors) from 2 to 100 in step sizes of 2, and for each k compute the average RMSE and average MAE obtained by averaging the RMSE and MAE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis) and average MAE (Y-axis) against k (X-axis).

The functions that might be useful for solving question 10 are described in the documentation below  
http://surprise.readthedocs.io/en/stable/knn_inspired.html  
http://surprise.readthedocs.io/en/stable/model_selection.html#surprise.model_selection.validation.cross_validate  
For question 10, use Pearson-correlation function as the similarity metric. You can read about how to specify the similarity metric in the documentation below:  
http://surprise.readthedocs.io/en/stable/similarities.html

#### Task 11 Use the plot from question 10, to ﬁnd a ’minimum k’. Note: The term ’minimum k’ in this context means that increasing k above the minimum value would not result in a signiﬁcant decrease in average RMSE or average MAE. If you get the plot correct, then ’minimum k’ would correspond to the k value for which average RMSE and average MAE converges to a steady-state value. Please report the steady state values of average RMSE and average MAE

#### Filter performance on trimmed test set
In this part of the project, we will analyze the performance of the k-NN collaborative ﬁlter in predicting the ratings of the movies in the trimmed test set. The test set can be trimmed in many ways, but we will consider the following trimming: 
1. Popular movie trimming: In this trimming, we trim the test set to contain movies that has received more than 2 ratings. To be speciﬁc, if a movie in the test set has received less than or equal to 2 ratings in the entire dataset then we delete that movie from the test set and do not predict the rating of that movie using the trained ﬁlter.   
2. Unpopular movie trimming: In this trimming, we trim the test set to contain movies that has received less than or equal to 2 ratings. To be speciﬁc, if a movie in the test set has received more than 2 ratings in the entire dataset then we delete that movie from the test set and do not predict the rating of that movie using the trained ﬁlter.   
3. High variance movie trimming: In this trimming, we trim the test set to contain movies that has variance (of the rating values received) of at least 2 and has received at least 5 ratings in the entire dataset. To be speciﬁc, if a movie has variance less than 2 or has received less than 5 ratings in the entire dataset then we delete that movie from the test set and do not predict the rating of that movie using the trained ﬁlter.  

Having deﬁned the types of trimming operations above, now we can evaluate the performance of the k-NN ﬁlter in predicting the ratings of the movies in the trimmed test set.  
#### Task 12 - Design a k-NN collaborative ﬁlter to predict the ratings of the movies in the popular movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of neighbors) from 2 to 100 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE

#### Task 13 -  Design a k-NN collaborative ﬁlter to predict the ratings of the movies in the unpopular movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of neighbors) from 2 to 100 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE  

#### Task 14 - Design a k-NN collaborative ﬁlter to predict the ratings of the movies in the high variance movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of neighbors) from 2 to 100 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE  

We provide you with the following hints that will help you solve questions 12,13, and 14: 
1. For each value of k, split the dataset into 10 pairs of training and test sets (trainset1,testset1),(trainset2,testset2),··· ,(trainset10,testset10) The following documentation might be useful for the splitting:   http://surprise.readthedocs.io/en/stable/getting_started.html#use-cross-validation-iterators   
2. For each pair of (trainset,testset): – Train the collaborative ﬁlter on the train set – Write a trimming function that takes as input the test set and outputs a trimmed test set – Predict the ratings of the movies in the trimmed test set using the trained collaborative ﬁlter – Compute the RMSE of the predictions in the trimmed test set   
3.  Compute the average RMSE by averaging across all the 10 folds

####  Performance evaluation using ROC curve
Receiver operating characteristic (ROC) curve is a commonly used graphical tool for visualizing the performance of a binary classiﬁer. It plots the true positive rate (TPR) against the false positive rate (FPR).  

In the context of recommendation systems, it is a measure of the relevance of the items recommended to the user. Since the observed ratings are in a continuous scale (0-5), so we ﬁrst need to convert the observed ratings to a binary scale. This can be done by thresholding the observed ratings. If the observed rating is greater than the threshold value, then we set it to 1 (implies that the user liked the item). If the observed rating is less than the threshold value, then we set it to 0 (implies that the user disliked the item). After having performed this conversion, we can plot the ROC curve for the recommendation system in a manner analogous to that of a binary classiﬁer  

#### Task 15  Plot the ROC curves for the k-NN collaborative ﬁlter designed in question 10 for threshold values [2.5,3,3.5,4]. For the ROC plotting use the k found in question 11. For each of the plots, also report the area under the curve (AUC) value.

For the ROC plotting, split the dataset into 90% for training and 10% for testing. For solving question 15 the functions described in the documentation below might be useful  
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

### Model-based collaborative ﬁltering
In model-based collaborative ﬁltering, models are developed using machine learning algorithms to predict users’ rating of unrated items. Some examples of model-based methods include decision trees, rule-based models, bayesian methods, and latent factor models. In this project, we will explore latent factor based models for collaborative ﬁltering.  

#### Latent factor based collaborative ﬁltering
Latent factor based models can be considered as a direct method for matrix completion. It estimates the missing entries of the rating matrix R, to predict what items a user will most probably like other than the ones they have rated. The basic idea is to exploit the fact that signiﬁcant portions of the rows and columns of the rating matrix are correlated. As a result, the data has builtin redundancies and the rating matrix R can be approximated by a low-rank matrix. The low-rank matrix provides a robust estimation of the missing entries.  
The method of approximating a matrix by a low-rank matrix is called matrix factorization. The matrix factorization problem in latent factor based model can be formulated as an optimization problem given by 4  
minimize U,V
m X i=1
n X j=1
(rij −(UV T)ij)2 (4)
In the above optimization problem, U and V are matrices of dimension m×k and n × k respectively, where k is the number of latent factors. However, in the above setting it is assumed that all the entries of the rating matrix R is known, which is not the case with sparse rating matrices. Fortunately, latent factor model can still ﬁnd the matrices U and V even when the rating matrix R is sparse. It does it by modifying the cost function to take only known rating values into account. This modiﬁcation is achieved by deﬁning a weight matrix W in the following manner:   
 Wij =(1,rij is known 0,rij is unknown
Then, we can reformulate the optimization problem as
minimize U,V
m X i=1
n X j=1
Wij(rij −(UV T)ij)2 (5)  
Since the rating matrix R is sparse, so the observed set of ratings is very small. As a result, it might cause over-ﬁtting. A common approach to address this problem is to use regularization. The optimization problem with regularization is given by equation 6. The regularization parameter λ is always non-negative and it controls the weight of the regularization term.
minimize U,V
m X i=1
n X j=1
Wij(rij −(UV T)ij)2 + λkUk2 F + λkVk2 F (6)
There are many variations to the unconstrained matrix factorization formulation (equation 6) depending on the modiﬁcation to the objective function and the constraint set. In this project, we will explore two such variations: 
* Non-negative matrix factorization (NNMF)  
* Matrix factorization with bias (MF with bias)

#### Non-negative matrix factorization (NNMF)

Non-negative matrix factorization may be used for ratings matrices that are non-negative. As we have seen in the lecture, that the major advantage of this method is the high level of interpretability it provides in understanding the useritem interactions. The main diﬀerence from other forms of matrix factorization is that the latent factors U and V must be non-negative. Therefore, optimization formulation in non-negative matrix factorization is given by 7  
minimize U,V
m X i=1
n X j=1
Wij(rij −(UV T)ij)2 + λkUk2 F + λkVk2 F subject to U ≥ 0,V ≥ 0
(7)  
There are many optimization algorithms like stochastic gradient descent (SGD),alternating least-squares (ALS),etc for solving the optimization problem in 7. Since you are familiar with the SGD method, so we will not describe it here. Instead, we will provide the motivation and main idea behind the ALS algorithm. SGD is very sensitive to initialization and step size. ALS is less sensitive to initialization and step size, and therefore a more stable algorithm than SGD. ALS also has a faster convergence rate than SGD. The main idea in ALS, is to keep U ﬁxed and then solve for V . In the next stage, keep V ﬁxed and solve for U. In this algorithm, at each stage we are solving a least-squares problem.  
Although ALS has a faster convergence rate and is more stable, but we will use SGD in this project. The main reason behind this is based on the fact that the python package that we will be using to design the NNMF-based collaborative ﬁlter only has the SGD implementation. This choice would have no eﬀect on the performance of the ﬁlter designed because both the SGD and ALS converges for the MovieLens dataset. The only downside of using SGD is that it will take a little bit longer to converge, but that will not be a big issue as you will see while designing the NNMF ﬁlter.  

#### Task 16 -  Is the optimization problem given by equation 5 convex? Consider the optimization problem given by equation 5. For U ﬁxed, formulate it as a least-squares problem.

#### Prediction function 

After we have solved the optimization problem in equation 7 for U and V , then we can use them for predicting the ratings.The predicted rating of user i for item j, denoted by ˆ rij, is given by equation 8  
ˆ rij = k X s=1 uis ·vjs (8)  
Having covered the basics of matrix factorization, now we are ready to implement a NNMF based collaborative ﬁlter to predict the ratings of the movies. We have provided you with the necessary background to implement the ﬁlter on your own, but we don’t require you to do that. Instead, you can use built-in functions in python for the implementation  

##### Design and test via cross-validation
In this part, you will design a NNMF-based collaborative ﬁlter and test it’s performance via 10-fold cross validation. Details on 10-fold cross validation have been provided in one of the earlier sections.

#### Task 17 - Design a NNMF-based collaborative ﬁlter to predict the ratings of the movies in the MovieLens dataset and evaluate it’s performance using 10-fold cross-validation. Sweep k (number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE and average MAE obtained by averaging the RMSE and MAE across all 10 folds. Plot the average RMSE (Y-axis) against k (X-axis) and the average MAE (Y-axis) against k (X-axis). For solving this question, use the default value for the regularization parameter.

For solving question 17, the functions described in the documentation below might be useful  
http://surprise.readthedocs.io/en/stable/matrix_factorization.html  

#### Task 18 - Use the plot from question 17, to ﬁnd the optimal number of latent factors. Optimal number of latent factors is the value of k that gives the minimum average RMSE or the minimum average MAE. Please report the minimum average RMSE and MAE. Is the optimal number of latent factors same as the number of movie genres?  

##### NNMF filter performance on trimmed test set

Having designed the NNMF ﬁlter in the previous section, now we will test the performance of the ﬁlter in predicting the ratings of the movies in the trimmed test set. We will use the same trimming operations as before

#### Task 19 -Design a NNMF collaborative ﬁlter to predict the ratings of the movies in the popular movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE  

#### Task 20 -Design a NNMF collaborative ﬁlter to predict the ratings of the movies in the unpopular movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE  

#### Task 21 - Design a NNMF collaborative ﬁlter to predict the ratings of the movies in the high variance movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE  

##### Performance Evaluation Using ROC Curve

In this part, we will evaluate the performance of the NNMF-based collaborative ﬁlter using the ROC curve. For details on the plotting of the ROC refer to the earlier sections.

#### Task 22  Plot the ROC curves for the NNMF-based collaborative ﬁlter designed in question 17 for threshold values [2.5,3,3.5,4]. For the ROC plotting use the optimal number of latent factors found in question 18. For each of the plots, also report the area under the curve (AUC) value.


##### Interpretability of NNMF 

The major advantage of NNMF over other forms of matrix factorization is not necessarily one of accuracy, but that of the high level of interpretability it provides in understanding user-item interactions. In this part of the project, we will explore the interpretability of NNMF. Speciﬁcally, we will explore the connection between the latent factors and the movie genres.

#### Task 23 - Perform Non-negative matrix factorization on the ratings matrix R to obtain the factor matrices U and V , where U represents the user-latent factors interaction and V represents the movie-latent factors interaction (use k = 20). For each column of V , sort the movies in descending order and report the genres of the top 10 movies. Do the top 10 movies belong to a particular or a small collection of genre? Is there a connection between the latent factors and the movie genres?

In task 23, there will be 20 columns of V and you don’t need to report the top 10 movies and genres for all the 20 columns. You will get full credit, as long as you report for a couple columns and provide a clear explanation on the connection between movie genres and latent factors.

#### Matrix Factorization with bias (MF with bias)

In MF with bias, we modify the cost function (equation 6) by adding bias term for each user and item. With this modiﬁcation, the optimization formulation of MF with bias is given by equation 9

##### Prediction function 
After we have solved the optimization problem in equation 9 for U,V,bu,bi, then we can use them for predicting the ratings. The predicted rating of user i for item j, denoted by ˆ rij is given by equation 10

##### Design and test via cross-validation

In this part, you will design a MF with bias collaborative ﬁlter and test it’s performance via 10-fold cross validation. Details on 10-fold cross validation have been provided in one of the earlier sections.

#### Task 24 -  Design a MF with bias collaborative ﬁlter to predict the ratings of the movies in the MovieLens dataset and evaluate it’s performance using 10-fold cross-validation. Sweep k (number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE and average MAE obtained by averaging the RMSE and MAE across all 10 folds. Plot the average RMSE (Y-axis) against k (X-axis) and the average MAE (Y-axis) against k (X-axis). For solving this question, use the default value for the regularization parameter.

For solving question 24, the function (SVD) described in the documentation below might be useful
http://surprise.readthedocs.io/en/stable/matrix_factorization.html  

#### Task 25 - Use the plot from question 24, to ﬁnd the optimal number of latent factors. Optimal number of latent factors is the value of k that gives the minimum average RMSE or the minimum average MAE. Please report the minimum average RMSE and MAE.  

##### MF with bias ﬁlter performance on trimmed test set
Having designed the MF with bias ﬁlter in the previous section, now we will test the performance of the ﬁlter in predicting the ratings of the movies in the trimmed test set. We will use the same trimming operations as before.

#### Task 26 - Design a MF with bias collaborative ﬁlter to predict the ratings of the movies in the popular movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE

#### Task 27 - Design a MF with bias collaborative ﬁlter to predict the ratings of the movies in the unpopular movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE

#### Task 28 -  Design a MF with bias collaborative ﬁlter to predict the ratings of the movies in the high variance movie trimmed test set and evaluate it’s performance using 10-fold cross validation.Sweep k ( number of latent factors) from 2 to 50 in step sizes of 2, and for each k compute the average RMSE obtained by averaging the RMSE across all 10 folds. Plot average RMSE (Y-axis) against k (X-axis). Also, report the minimum average RMSE  


##### Performance evaluation using ROC Curve

In this part, we will evaluate the performance of the MF with bias collaborative ﬁlter using the ROC curve. For details on the plotting of the ROC refer to the earlier sections.

#### Task 29 Plot the ROC curves for the MF with bias collaborative ﬁlter designed in question 24 for threshold values [2.5,3,3.5,4]. For the ROC plotting use the optimal number of latent factors found in question 25. For each of the plots, also report the area under the curve (AUC) value.

##  Naive collaborative ﬁltering
In this part of the project, we will implement a naive collaborative ﬁlter to predict the ratings of the movies in the MovieLens dataset. This ﬁlter returns the mean rating of the user as it’s predicted rating for an item.

### Prediction function 

The predicted rating of user i for item j, denoted by ˆ rij is given by equation 11
  rij = µi (11)
 where µi is the mean rating of user i.

### Design and test via cross-validation 
Having deﬁned the prediction function of the naive collaborative ﬁlter, we will design a naive collaborative ﬁlter and test it’s performance via 10-fold cross validation.  

#### Task 30 - Design a naive collaborative ﬁlter to predict the ratings of the movies in the MovieLens dataset and evaluate it’s performance using 10-fold cross validation. Compute the average RMSE by averaging the RMSE across all 10 folds. Report the average RMSE  
An important thing to note about the naive collaborative ﬁlter is that there is no notion of training. For solving question 30, split the dataset into 10 pairs of train set and test set and for each pair predict the ratings of the movies in the test set using the prediction function (no model ﬁtting required). Then compute the RMSE for this fold and repeat the procedure for all the 10 folds. The average RMSE is computed by averaging the RMSE across all the 10 folds.  

#### Naive collaborative ﬁlter performance on trimmed test set

Having designed the naive collaborative ﬁlter in the previous section, now we will test the performance of the ﬁlter in predicting the ratings of the movies in 13 the trimmed test set. We will use the same trimming operations as before.

#### Task 31 - Design a naive collaborative ﬁlter to predict the ratings of the movies in the popular movie trimmed test set and evaluate it’s performance using 10-fold cross validation. Compute the average RMSE by averaging the RMSE across all 10 folds. Report the average RMSE.

#### Task 32 - Design a naive collaborative ﬁlter to predict the ratings of the movies in the unpopular movie trimmed test set and evaluate it’s performance using 10-fold cross validation. Compute the average RMSE by averaging the RMSE across all 10 folds. Report the average RMSE.

#### Task 33 - Design a naive collaborative ﬁlter to predict the ratings of the movies in the high variance movie trimmed test set and evaluate it’s performance using 10-fold cross validation. Compute the average RMSE by averaging the RMSE across all 10 folds. Report the average RMSE.

## Performance Comparison 
In this section, we will compare the performance of the various collaborative ﬁlters (designed in the earlier sections) in predicting the ratings of the movies in the MovieLens dataset.

#### Task 34 Plot the ROC curves (threshold = 3) for the k-NN, NNMF, and MF with bias based collaborative ﬁlters in the same ﬁgure. Use the ﬁgure to compare the performance of the ﬁlters in predicting the ratings of the movies.

## Ranking
Two primary ways in which a recommendation problem may be formulated are:
1. Prediction version of problem: Predict the rating value for a user-item combination
2. Ranking version of problem: Recommend the top k items for a particular user
In previous parts of the project, we have explored collaborative ﬁltering techniques for solving the prediction version of the problem. In this part, we will explore techniques for solving the ranking version of the problem. There are two approaches to solve the ranking problem: 
* Design algorithms for solving the ranking problem directly 
* Solve the prediction problem and then rank the predictions 

Since we have already solved the prediction problem, so for continuity we will take the second approach to solving the ranking problem.

### Ranking predictions

The main idea of the second approach is that it is possible to rank all the items using the predicted ratings. The ranking can be done in the following manner: 
* For each user, compute it’s predicted ratings for all the items using one of the collaborative ﬁltering techniques. Store the predicted ratings as a list L. 
* Sort the list in descending order, the item with the highest predicted ratings appears ﬁrst and the item with the lowest predicted ratings appears last. 
* Select the ﬁrst t-items from the sorted list to recommend to the user.

### Evaluating ranking using precision-recall curve

Precision-recall curve can be used to evaluate the relevance of the ranked list. Before stating the expressions for precision and recall in the context of ranking, let’s introduce some notation:  
   S(t) : The set of items of size t recommended to the user. In this recommended set, ignore (drop) the items for which we don’t have a          ground truth rating.
    G: The set of items liked by the user (ground-truth positives)
Then with the above notation, the expressions for precision and recall are given by equations 8 and 9 respectively


 Precision(t) = |S(t)∩G|/|S(t)|
 Recall(t) = |S(t)∩G|/|G| 
 
 #### Task 35- Precision and Recall are deﬁned by the mathematical expressions given by equations 12 and 13 respectively. Please explain the meaning of precision and recall in your own words.
 
 Both precision and recall are functions of the size of the recommended list (t). Therefore, we can generate a precision-recall plot by varying t.

#### Task 36 - Plot average precision (Y-axis) against t (X-axis) for the ranking obtained using k-NN collaborative ﬁlter predictions. Also, plot the average recall (Y-axis) against t (X-axis) and average precision (Y-axis) against average recall (X-axis). Use the k found in question 11 and sweep t from 1 to 25 in step sizes of 1. For each plot, brieﬂy comment on the shape of the plot.

#### Task 37 - Plot average precision (Y-axis) against t (X-axis) for the ranking obtained using NNMF-based collaborative ﬁlter predictions. Also, plot the average recall (Y-axis) against t (X-axis) and average precision (Y-axis) against average recall (X-axis). Use optimal number of latent factors found in question 18 and sweep t from 1 to 25 in step sizes of 1. For each plot, brieﬂy comment on the shape of the plot.
#### Task 38 - Plot average precision (Y-axis) against t (X-axis) for the ranking obtained using MF with bias-based collaborative ﬁlter predictions. Also, plot the average recall (Y-axis) against t (X-axis) and average precision (Y-axis) against average recall (X-axis). Use optimal number of latent factors found in question 25 and sweep t from 1 to 25 in step sizes of 1. For each plot, brieﬂy comment on the shape of the plot.

We provide you the following hints that will help you solve questions 36,37, and 38: 
* Use threshold = 3 for obtaining the set G   
* Use 10-fold cross-validation to obtain the average precision and recall values for each value of t. To be speciﬁc, compute precision and recall for each user using equations 12 and 13 and then average across all the users in the test set to obtain the precision and recall for this fold. Now repeat the above procedure to compute the precision and recall for all the folds and then take the average across all the 10-folds to obtain the average precision and average recall value for this value of t.  
* If |G| = 0 for some user in the test set, then drop this user  
* If some user in the test set has rated less than t items, then drop this user 


#### Task 29 - Plot the precision-recall curve obtained in questions 36,37, and 38 in the same ﬁgure. Use this ﬁgure to compare the relevance of the recommendation list generated using k-NN, NNMF, and MF with bias predictions.


 

















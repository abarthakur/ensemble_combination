# Classifier Ensemble with Data Dependent Classifier Fusion

## Project Overview



## Problem Statement

* Given a classification task with a labelled dataset split into a training and test set and an ensemble of heterogeneous classifiers trained on some subset of the training set, our objective is to design a classifier fusion rule to combine their predictions on a novel sample (from the test set). 
* The fusion rule should be data-dependent. 
	* Data independent rules like voting, average, product etc derive the final prediction of the ensemble purely from the output of the individual classifiers.
	* [Decision Templates][2] proposed by Kuncheva et al in 2001, are an example of a rule that takes the data point into account.
* The overall performance of the ensemble on the test set should be better than the best performance of any individual classifier.


### Notation

* Let T denote the training set.
* Let L be the number of classifiers.
* Let M be the number of classes.
* Let each classifier output M values representing the posterior probabilities \[P(Y=1|X),P(Y=2|X)...P(Y=M|X)\].


Let the set of classifiers be H = { h_1, h_2, . .. ..  h_L } where L is the number of classifiers and set of classes be C = { c_1, c_2, ..  c_M } where M is the number of classes. Let D be the labelled dataset.  The output of classifier h_i for input instance x is represented by h_i(x) = { h_i^1(x), h_i^2(x), .. h_i^M(x) } where h_i^j(x) represent the class posterior probability of classifier h_i for class c_j for input instance x.


## Preliminaries

Our proposed approach is built on the concepts of the [K-Nearest Neighbours][1] classifier and the [Decision Template][2] combination rule for ensembles. We thus begin with a brief description of both.

### K-Nearest Neighbours Algorithm

* Given a sample x, and a database of samples X, the k-nearest neighbours algorithm identifies the k points "n" in X such that distance(x,n) is minimized.
	* The distance measure is the most important component which defines how points are similar. Cosine distance and Euclidean distance are simple off-the-shelf choices.
* When trying to estimate some unknown property of a novel sample (which may be a discrete or continuous value), the underlying assumtion is that the required value can be determined from the samples in the training set (for whom this property is known) which are "closest" or most "similar" to the novel sample.
* A simple classification model based on KNN finds the neighbours of a novel sample, and combines their labels using majority voting. Another layer of sophistication can be weighted ajority voting, where closer neighbours contribute more to the final result.

### Decision Templates for Classifier Fusion

* A decision profile of an instance x, denoted DP(x), is an LxM matrix such that
> DP(X)<sub>i,j</sub>= P<sub>i</sub>(Y=j|X) , where P<sub>i</sub> is the posterior probability distribution of the i'th classifier in the ensemble.
* A decision template of a class ``c``, denoted DT<sub>c</sub> is the mean of the decision profiles of positive samples of class ``c`` in the training set.
* Thereafter, given a novel sample ``x``, it is assigned the class according to the rule
> predicted = argmax<sub>c</sub> similarity(DP(x),DT<sub>c</sub>)
* Here similarity is used in a vague sense and the author proposes a number of candidates based on fuzzy logic.

## Proposed Approach

We choose to extend the [Decision Template][2] method which is already a data independent method - and combine it with the idea of k-Nearest Neighbours in 3 different ways.

### Approach 1 : Local Decision Templates

* *Preprocessing* : We build a nearest-neighbours model over the training set T. 
	* Let us denote this model as NN, such that 
	> NN(t)={r_1,r_2,r_3,...r_k} where t,r_i &isin; T.
* For a novel sample x, we compute DP(x). We then compute ``NN(x)``.
* We then compute decision templates for each class from ``NN(x)`` (as opposed to T). If a class has no samples in ``NN(x)``, we set its decision templates entries to ``inf``.
* Our final classification rule remains the same
> predicted = argmax<sub>c</sub> similarity(DP(x),DT<sub>c</sub>)

### Approach 2 : Fine-grained Decision Templates

* *Preprocessing* : For all points in the training set we compute their decision profiles and build a nearest neighbour model with Euclidean distance as the chosen metric. 
	* Let us denote this model as NNDP, such that 
	> NNDP(DP(t))={r_1,r_2,r_3,...r_k} where t,r_i &isin; T.
	* Thus NNDP takes as input a decision profile and returns those samples in T whose decision profiles are close.
* For a novel sample x, we compute DP(x). We then compute ``NNDP(DP(x))`` (as opposed to T).
* Our final classification rule remains the same.

### Approach 3 : Decision Template based Weighted Voting

* *Preprocessing* : We build a k-nearest neighbours model over decision profiles, denoted ``NNDP(DP(t))``, just like Approach 2. 
* Given a novel sample x, we compute DP(x). We then compute ``NNDP(DP(x))`` (as opposed to T).
* For each classifier, we score it according to **how many samples it correctly classifies in NNDP(DP(x))**. A simple score, s_i, is the count itself which is what we have used.
* Then we choose our predicted class according to a weighted voting rule, where the i'th classifier contributes s_i to the class it predicts. That is,
> output =  &sum;<sub>0</sub><sup>L</sup> argmax<sub>c</sub>  s_i * I[ Pred<sub>i</sub> = c] \
> where I is an indicator variable, and Pred<sub>i</sub> is the predicted class of the classifier, usually - \
>  Pred<sub>i</sub> = argmax<sub>c'</sub> P<sub>i</sub>(Y=c'|x)

## Evaluation Parameters

* We evaluated these approaches on the LandSat dataset, using a 8 unit single layer neural network as the base classifer.

## Credits

* This project was completed as part of the requirements for Data Mining course taught by Dr. Amit Awekar in 2017
* Group Members - Aneesh Barthakur, Prateek Vij, Vaibhav Saxena, Uppinder Chugh

## References

[1]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm  "Wikipedia : k-nearest neighbors algorithm"
[2]: https://www.sciencedirect.com/science/article/abs/pii/S003132039900223X "Decision templates for multiple classifier fusion: an experimental comparison : Kuncheva et al 2001"

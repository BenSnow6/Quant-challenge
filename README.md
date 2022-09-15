# QRT 2022 Quant Challenge

## Introduction

In this I will be investigating the [Qube Research 2022 Data Challenge](https://challengedata.ens.fr/participants/challenges/72/).

First, it is of great importance to me to describe this problem for technical and non-technical people alike. I will first attempt to explain this problem in a way that is accessible to all and then, for the mathematically inclined, I will provide a more rigorous description of the problem.

## Non technical problem definition

Everyone wants to make money in the stock market and the best way to do this is to buy low and sell high. The problem is that it is very difficult to know when to buy and when to sell. One idea is to use historical data from many stocks to predict the price in the future. As you can imagine, this is not easy or else everyone would do it and make lots of money. As such, we need to think carefully about how to do it and set reasonable expectations for our predictive abilities.

We start with the past three years' returns of a a group of $N$ stocks as our training data. Some of the information contained in this dataset will be useful in predicting the future returns of the stocks, but some of it will be useless. For example, imagine a we are trying to predict the change in Microsoft stock in the next week. In our dataset we have information about a range of stocks from lots of industries: Utilities, healthcare, financial, technology, etc. The information about the change in the stock price of a utility company may not be too useful in predicting the change in the stock price of Microsoft. In fact, it is could even be harmful to our predictive ability. Other companies similar to Microsoft, such as Google, Apple, and Amazon, may be more useful. We therefore need to find a way of selecting the most useful information from our dataset to find the best predictors of the future returns of all of the stocks.

## Technical problem definition

The problem is defined as follows:

Using a stock market of *N* stocks with returns $R_t \in \mathbb{R}^N$ at time *t* design a vector $S_{t+1} \in \mathbb{R}^N$ from the information such that the prediction overlap $\langle S_{t+1} , \mathbb{R}_{t+1} \rangle$ is positive and maximised.

In other words, can we use the returns of *N* stocks to predict the returns of the same stocks in the future?

Initially, the problem is approached by using a simple linear factor model to learn factors (aka features) of a non-linear parameter space. Using linear estimators, a parametric model of the form

$$
S_{t+1} =  \sum_{l=1}^F \beta_l F_{t,l}
$$

where the vectors $F_{t,l} \in \mathbb{R}^N$ are the factors produced by financial experts and $\beta_l,...,\beta_F \in \mathbb{R}$ are the model parameters, is fit to a training dataset.

The exact form of these factors is not known due to the assumption of linear factor models. That is, they are assumed to be unmeasurable. In the finance world, examples of factors used by investors are the 5-day normalised mean returns $R_{t}^{(5)}$ and the Momentum $M_t := \frac{1}{\sqrt{m}} \sum_{k=1}^m R_{t+1-k}$. It is thought that a combination of these factors, along with other features, can be used to predict the returns of stocks. If we don't have any idea about finance then we can instead make assumptions about the form of the factors.

We will start by assuming that the features are linear combinations of the past returns of the stocks. That is, we will assume that the factors are of the form:

$$
F_{t,l} = \sum_{k=1}^D A_{k,l} R_{t+i-k}
$$

for some basis vectors $A_l := (A_{kl}) \in \mathbb{R}^{D}$ and a fixed time depth $D$ days. Very much like an eigen-decomposition problem or a multidimensional fourier decomposition, we will hold that the basis vectors are orthonormal. That is, $\langle A_l, A_k \rangle = \delta_{lk}  \forall l,k$. This is nice since we can now use physics intuition to understand the problem better.


### Physics intuition
It is common in physics to view a problem in terms of a linear combination of basis vectors. Examples of this are the fourier decomposition of a signal, the eigen-decomposition of a matrix, and the spherical harmonics. In each of these cases, the basis vectors are orthonormal and the coefficients are the parameters of the problem. In the case of the fourier decomposition, the coefficients are the amplitudes of the sinusoids. In the case of the eigen-decomposition, the coefficients are the eigenvalues. In the case of the spherical harmonics, the coefficients are composed of the Legendre polynomials. The idea of decomposing a problem into a set of basis functions is made more powerful by enforcing an orthoginality condition. Orthonormal basis functions are normalised and orthogonal to each other. The very simplest set of orthonormal basis functions are the unit vectors $\mathbf{e}_l \in \mathbb{R}^N$, explicitly:
$$
\mathbf{e}_1 = (1,0,...,0), \mathbf{e}_2 = (0,1,...,0), ..., \mathbf{e}_N = (0,0,...,1).
$$

Each of these basis vectors is orthogonal to all of the others and has a norm of 1. This means that we can use these basis vectors to represent any vector in $\mathbb{R}^N$ as a linear combination of the basis vectors. For example, the vector $\mathbf{v} = (1,2,3,4,5)$ can be represented as a linear combination of the basis vectors as follows:

$$
\mathbf{v} = \mathbf{e}_1 + 2\mathbf{e}_2 + 3\mathbf{e}_3 + 4\mathbf{e}_4 + 5\mathbf{e}_5
$$

All simple stuff so far. The power of this intuition is to map the linear factor model idea to that of something we already have a good intuition for. In the case of the linear factor model, we are trying to design a set of orthonormal basis vectors $A_{k,l} \in \mathbb{R}^D$ that can be used with the returns $R_{t} \in \mathbb{R}^N$ to predict the returns $S_{t+1} \in \mathbb{R}^N$. We find $\beta_l$ by minimising the mean squared error between the predicted returns $S_{t+1}$ and the actual returns $\mathbb{R}_{t+1}$. Ok, we'll look at the example notebook to get an idea of how to implement this in practice ad then we'll come back to the explaination.
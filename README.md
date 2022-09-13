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
It is common for problems in physics to involve finding the basis functions of a given function, an example being [Fourier decomposition](https://en.wikipedia.org/wiki/Fourier_decomposition). This takes a periodic function and decomposes it into a sum of sine and cosine functions.
A reminder of an example of this is approcimating the sawtooth function:

$$
s(x) = \frac{x}{\pi}, \; \text{for} \; -\pi < x < \pi,
$$
$$
s(x + 2\pi k) = s(x) \; \text{for} -\pi < x < \pi \; \text{and} \; k \in \mathbb{Z} 

$$
This summation can be written as  This is a very useful tool in physics and is used to solve many problems. In the same way, we can use the basis vectors $A_l$ to decompose the factors $F_{t,l}$ into a sum of simpler functions. This is a very useful tool in finance and is used to solve many problems.
# QRT 2022 Quant Challenge

## Introduction

In this I will be investigatin the [Qube Research 2022 Data Challenge](https://challengedata.ens.fr/participants/challenges/72/).

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

where the vectors $F_{t,l} \in \mathbb{R}^N$ are the factors produced by financial experts and $\beta_l,...,\beta_F \in \mathbb{R}$ are the model parameters that can be fit to a training dataset.




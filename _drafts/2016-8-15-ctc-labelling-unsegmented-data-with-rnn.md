---
layout: post
title: "Connectionist Temporal Classification - Labelling Unsegmented Sequence Data with Recurrent Neural Networks (Paper)"
comments: true
excerpt: By Alex Graves, Santiago Fernández, Faustino Gomez, Jurgen Schmidhuber. ICML, 2006
---

**by** Alex Graves, Santiago Fernández, Faustino Gomez, Jurgen Schmidhuber. ICML, 2006. [[Link]](http://www.cs.toronto.edu/~graves/icml_2006.pdf){:target="_blank"}

**Observation**: This paper was a major breakthrough on its time, removing the confusing pipeline that exists in hybrid approaches like GMM/HMM, DNN/HMM. This paper obtained a reduction of 3% LER on TIMIT dataset, outperforming the state-of-arts methods.

![Comparison against others methods](/assets/graves2006/comparison.png){:class="img-responsive img-center" style="max-width: 300px;"}

**Insight**: The key idea behind the CTC method is to interpret the network outputs as a probability distribution over all possible label sequences, given the input sequence. Given this distribution, an objective function can be derived that directly maximizes the probabilities of the correct labeling.

## New type of algorithms: end-to-end recognizers

Deep learning is conquering as the state-of-art of several algorithms, such as image classification, image segmentation, sentiment analysis, machine translation, and speech recognition. In image task areas, deep learning methods are applied directly to raw input images, outperforming several engineering features, such as the use of HOG to image detection.
Unfortunately, the same cannot be said to speech recognition area. Automatic speech recognition has greatly benefited from the introduction of neural network and unsupervised learning, although, has been applied only as a single component in a complex pipeline. The first step of pipeline is input feature extraction: standard techniques include mel-scale filter banks (with or without a further transform into Cepstral coefficients) and speaker normalization techniques such as vocal tract length normalization. Neural networks are then trained to classify individual frames of acoustic data, and their output distributions are reformulated as emission probabilities for a HMM. The objective function used to train the network are a quite different from the true performance measure (sequence-level transcription accuracy). This is precisely the sort of inconsistent that end-to-end learning seeks to avoid. Over the years, researchers founded that a large gain in frame accuracy could not translate to transcription accuracy, even could degraded. Thus, building state-of-art ASR systems remains a complicated, expertise-intensive task (dictionaries, phonetic questions, segmented data, GMM models to obtain initial frame-level labels, multiples stages with different features processing techniques, an expert to determine the optimal configurations of a multitude of hyper-parameters, and so on). Why not develop a system applying neural networks as a single pipeline?
The first successful shot was made by Alex Graves et. al. in 2006, proposing the Connectionist Temporal Classification (CTC) in “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks“. The main goal of his method is that CTC was specifically designed for temporal classification tasks; that is, for sequence labelling problems where the alignment between the inputs and the target labels is unknown. It also does not require pre-segmented training data, or external post-processing to extract the label sequence from the network outputs.
Since 2006,  the CTC method has been extensively used in end-to-end speech recognition systems, and was adopted by Google [last year]( https://research.googleblog.com/2015/09/google-voice-search-faster-and-more.html){:target="_blank"}.


## Connectionist Temporal Classification

The CTC consists of a softmax output layer, with one more unit than there are labels. The activations of softmax indicates the probabilities of outputting the corresponding labels at particular time, given the input sequence and the network weights. Considering \\(K\\) unique labels, the neural network output will emit probabilities of \\(K+1\\) where the extra label will give a probability of outputting a *blank*, or no label.

Given an utterance \\(\mathbf{X} = (\mathbf{x}_1,\ldots,\mathbf{x}_T)\\), its label sequence is denoted as \\(\mathbf{z} = (z_1,\ldots,z_U)\\), where the *blank* label \\(\emptyset\\) will be indexed as \\(0\\). Therefore, \\(z_u\\) is a integer ranging from \\(1\\) to \\(K\\). The length of \\(\mathbf{z}\\) is constrained to be no greater than the length of the utterance, i.e, \\(U \leq T\\). CTC aims to maximize \\( \log \Pr(\mathbf{z}\|\mathbf{X}) \\), the log-likelihood of the label sequence given the inputs, by optimizing the RNN model parameters.

The final layer of the RNN is a softmax layer which has \\(K + 1\\) nodes that correspond to the \\(K + 1\\) labels (including \\(\emptyset\\)). At each frame \\(t\\), we get the output vector \\(\mathbf{y}_t\\), whose \\(k\\)-th element \\(y_t^k\\) is the posterior probability of the label \\(k\\).

Then, if we assume the output probabilities at each time step to be independent given \\(\mathbf{X}\\), we get the following conditional distribution:

$$
\Pr(\mathbf{p}|\mathbf{X}) = \prod_{t=1}^{T} y_t^{p_t},
$$

where \\(\mathbf{p} = (p_1,\ldots, p_T)\\) is the CTC path, a sequence of labels at frame level. It differs from \\(\mathbf{z}\\) in that the CTC path allows the occurrences of the blank label and repetitions of non-blank labels. The label sequence \\(\mathbf{z}\\) can the be mapped to its corresponding CTC paths. This is a one-to-multiple mapping because multiple CTC paths can correspond to the same label sequence, e.g., both "A A \\(\emptyset\\) B C \\(\emptyset\\)" and  "\\(\emptyset\\) A A B \\(\emptyset\\) C C" are mapped to the label sequence "A B C". Considering the set of CTC paths for \\(\mathbf{z}\\) as \\(\Phi(\mathbf{z})\\), the likelihood of \\(\mathbf{z}\\) can be evaluated as a sum of the probabilities of its CTC paths:

$$
\Pr(\mathbf{z}|\mathbf{X}) = \sum_{\mathbf{p} \in \Phi(\mathbf{z})} \Pr(\mathbf{p}|\mathbf{X}).
$$

This is the core of CTC. Doing this ‘collapsing together’ of different paths onto the same label sequence is what makes it possible for CTC to use unsegmented data, because it allows the network predict the labels without knowing where they occur. In practice, CTC tends to output labels close to where they occur in the input sequence.

As we can see, summing over all the CTC paths is computationally impractical. A solution is to represent the possible CTC paths compactly as a trellis. To allow blanks in CTC paths, we add “0” to beginning and the end of \\(\mathbf{z}\\), and also insert “0” between every pair of the original labels in \\(\mathbf{z}\\). The resulting augmented label sequence \\(\mathbf{l} = (l_1,\ldots,l_{2U+1})\\) is leverage in a forward-backward algorithm for efficient likelihood evaluation. In the forward pass, \\(\alpha(t,u)\\) represents the total probability of all CTC paths that end with label \\(l_u\\) at frame \\(t\\). As with the case of HMMs, \\(\alpha(t,u)\\) can be recursively computed from the previous states. Similarly, a backward variable \\(\beta(t,u)\\) carries the total probability of all CTC paths that starts with label \\(l_u\\) at time \\(t\\) and reaches the final frame \\(T\\) and can be computed recursively from the next states. So, defining the set \\(\varphi(t,u) = \\{\mathbf{p}_t = (p_1, \ldots, p_t)~:~\mathbf{p_t} \in \Phi(\mathbf{z}\_{u/2}), p_t = l_u\\} \\)

$$
    \alpha(t,u) = \sum_{\mathbf{p}_t \in \varphi(t,u)} \prod_{i=1}^t y_t^{p_i}.
$$

Given the above formulation is easy to see that the probability if \\(\mathbf{z}\\) can be expressed as the sum of the forward variables with and without the final blank at time \\(T\\)

$$
\Pr(\mathbf{z}|\mathbf{X}) = \alpha(T, 2U+1) + \alpha(T, 2U).
$$

Since all paths must start with either a \\(\emptyset\\) or the first symbol in \\(\mathbf{z}\\), \\(z_1\\), we have the following initial conditions:

$$
\begin{align}
\alpha(1,1) &= y^{\emptyset}_1 \\
\alpha(1,2) &= y^{l_2}_1 \\
\alpha(1,u) &= 0,\ \forall u > 2.
\end{align}
$$

For \\(t=2\\) we have:

$$
\begin{align}
\alpha(2,1) &= y_2^{l_1} \\
\alpha(2,2) &= (\alpha(1,1) + \alpha(1,2))y_2^{l_2} \\
\alpha(2,3) &= \alpha(1,2)y_2^{l_3} \\
\alpha(2,4) &= \alpha(1,2)y_2^{l_4}\\
\alpha(2,u) &= 0,\ \forall u > 4,
\end{align}
$$

for \\(t=3\\)

$$
\begin{align}
\alpha(3,1) &= (\alpha(1,1) + \alpha(2,1))y_3^{l_1} \\
\alpha(3,2) &= (\alpha(2,1) + \alpha(2,2))y_3^{l_2} \\
\alpha(3,3) &= (\alpha(2,2) + \alpha(2,3))y_3^{l_3} \\
\alpha(3,4) &= (\alpha(2,2) + \alpha(2,3) + \alpha(2,4))y_3^{l_4}\\
\alpha(3,5) &= \alpha(2,4)y_3^{l_5} \\
\alpha(3,6) &= \alpha(2,4)y_3^{l_6} \\
\alpha(3,u) &= 0\ \forall u > 6.
\end{align}
$$

If we go on with the trellis (see figure below)

![Trellis of CTC algorithm](/assets/ctc_forward-backward.png){:class="img-responsive img-center" style="max-width: 350px;"}

we can find the following recursion

$$
   \alpha(t,u) = y^{l_u}_t \sum_{i=f(u)}^u \alpha(t-1, i),
$$

where

$$
f(u) = \begin{cases}
u-1 & \text{if } l_u = \emptyset \text{ or } l_{u-2} = l_u \\
u-2 & \text{otherwise},
\end{cases}
$$

with the boundary condition

$$
\alpha(t,0) = 0\ \forall t.
$$

We can see that

$$
\alpha(t,u) = 0\ \forall u < 2U+1 - 2(T-t) - 1,
$$

because these variables correspond to states for which there are not enough timesteps left to complete the sequence.

Doing the same approach to \\(\beta(t,u)\\) and defining the set \\(\varrho(t,u) = \\{\mathbf{p}\_{T-t}: (\mathbf{p}\_{T-t} \cup \mathbf{p}\_t) \in \Phi(\mathbf{z}),\ \forall \mathbf{p}_t \in \varphi(t,u)\\}\\) we have the following initial conditions:

$$
\begin{align}
\beta(T, 2U+1) &= \beta(T, 2U + 1 - 1) = 1 \\
\beta(T, u) &= 0,\ \forall u < 2U+1 - 1.
\end{align}
$$

Calculating \\(\beta(t,u)\\) we have

$$
\beta(t,u) = \sum_{i=u}^{g(u)} \beta(t+1, i) y_{t+1}^{l_i},
$$

where

$$
g(u) =  \begin{cases}
u+1 & \text{if } l_u = \emptyset \text{ or } l_{u+2} = l_u \\
u+2 & \text{otherwise},
\end{cases}
$$

We can see that

$$
\beta(t,u) = 0,\ \forall u > 2t,
$$

with the boundary conditions

$$
\beta(t,2U+1 + 1) = 0\, \forall t.
$$

### Loss Function
The likelihood of sequence \\(\mathbf{z}\\) can then be computed as:

$$
\Pr(\mathbf{z}|\mathbf{X}) = \sum_{u=1}^{2U+1} \alpha(t,u)\beta(t,u),
$$

where t can be any frame \\(1\leq t \leq T\\).

#### Loss Gradient
The objective \\(\log \Pr(\mathbf{z}|\mathbf{X})\\) now becomes differentiable with respect to the RNN outputs \\(\mathbf{y}_t\\). Defining an operation on the augmented label sequence \\(\Gamma(\mathbf{l}, k) = \\{u|l_u=k\\} \\) that returns the elements of \\(\mathbf{l}\\) which have the value \\(k\\). The derivative of the objective with respect to \\(y_t^k\\) can be derived as:

$$
\frac{\partial \log \Pr(\mathbf{z}|\mathbf{X})}{\partial y_t^k} = \frac{1}{\Pr(\mathbf{z}|\mathbf{X})}\frac{1}{y_t^k} \sum_{u \in \Gamma(\mathbf{l}, k)} \alpha(t,u) \beta(t,u)
$$

## Experiments

Now that we have a general idea behind the CTC method we'll try to reproduce some experiments of the paper.

### Evaluation

Before starting showing network topologies and parameters, we have to define our figure of merit. In the paper, the authors applied the Label Error Rate, which is the mean normalised edit distance between its classifications and the targets in the test set $$S$$.

$$
\mathrm{LER}(h, S) = \frac{1}{|S|} \sum_{(\mathbf{x},\mathbf{z}) \in S} \frac{\mathrm{ED}(h(\mathbf{x}), \mathbf{z})}{|\mathbf{z}|},
$$

where $$\mathrm{ED}(\mathbf{p}, \mathbf{q})$$ is the edit distance between two sequences $$\mathbf{p}$$ and $$\mathbf{q}$$ ([Wikipedia](https://en.wikipedia.org/wiki/Edit_distance)).

### Data

The authors used the TIMIT dataset. If you are not familiar with this dataset, please see my [another]({% post_url 2016-7-19-ctc-tensorflow-timit %}) post for more information.

The audio data was preprocessed using a frame signal, $$L$$, of 10 ms, with 5 ms of overlapping $$H$$, using 12 Mel-Frequency Cepstrum Coefficients (MFCCs) from 26 filter-bank channels. The log-energy was also added, plus the first derivatives, giving a vector of 26 coefficients per frame.

<div class="table-wrapper center" markdown="block">

Parameter|Value
---|---|
L|10 ms
H|5 ms
num_coeff|13
num_fbanks|26
energy|True
d|True
dd|False
 ---|---|

</div>

 where `num_coeff` represents the number of MFCC coefficient to be used (including the 0th), `num_fbanks` is the number of filter-bank channels, `energy` determines when the 0th coefficient will be replaced by the energy of framed signal, `d` indicates if the first derivative willbe used, and `dd` indicates if the second derivative will be used.

 **Instead of using the complete test set** as was proposed by the paper, here we'll use the core test set, because the results will be more reliable and unbiased, and 400 utterances of complete test set will be employed as validation data.

 As a target, they used 61 phonemes (+ blank label) that came from TIMIT dataset transcriptions.

### Network topology

Graves *et. al* used a **bidirectional LSTM** (you can read an amazing post about LSTM networks [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)) with **100 memory blocks** in each direction and **peepholes connections**, followed by **fully connection through time** (*i.e.*, sharing weights across each timestep), and a **sotfmax layer** with **CTC cost function**. Plus, they added a random **gaussian noise layer after the input** with **standard deviation of `0.6`** to improve generalization, as you can see in the image below:

![Trellis of CTC algorithm](/assets/graves2006/topology.svg){:class="img-responsive img-center"}

#### Training parameters
All **weights** were initialized with **uniform random noise from -0.1 to 0.1** and the forget bias were kept at zero. The **stochastic gradient descent method** was used with **learning rate of 10e-4** and a **momentum** of **0.9** in a **batch size of one** sample.

### Results

First of all, using a batch size of 1 is time consuming. I ran my simulations on my lab's mid-end machines, only on CPU, and each forward inference and backpropagation trough time took about 21 minutes. Considering 200 epochs, each simulation will take about 70 hours, almost 3 days. Yeah, this is bad, because the dataset is tiny compared to others and we can spend several weeks fine tunning our hyper-parameters. So, I also ran simulations with bigger batch size to see with my own eyes which size is better.

#### Increasing the batch size

Unfortunately, I had no time to run my simulations several times, so my results are for only one simulation for each batch size. This is not so bad, because as far as I noticed, there aren't a giant deviation through several runs. I ran the same model as cited above, but with a variable `batch_size = {1, 16, 32, 64, 256}` and the results you can see below.

![Unormalized loss](/assets/graves2006/loss_batch_size.svg){:class="img-responsive img-center"}

First of all, this graphic is not fair. Despite the batch size is increasing, we have a pitfall; we do less gradient updates in each epoch, but in the same time, we enjoy that each epoch is faster with bigger batch size (due to the use of larger matrix and therefore better use of the computational time of the CPU). Calculating the time spent in each epoch and calculating the median we can get the following speedups:

Batch Size|Speedup|Median (min)
---|---|---|
1|1.00x|21.86
16|2.03x|10.78
32|2.37x|9.24
64|2.73x|7.99
256|2.94x|7.43
 ---|---|---|

and normalizing  the x-axis by the speedup we get the following figure:

![Normalized loss](/assets/graves2006/loss_norm_batch_size.svg){:class="img-responsive img-center"}

As you can see, using a bigger batch size we get a better gradient estimation, so the cost evolves more smoother than using a batch size of 1. But, as we increase our batch size we minimize the network to the directions that are close to each other, and we can get the network stuck at poor minimas. In the other hand, using batch size of length 1, our training could be too slow to converge to one minima. For this dataset and network topology, the best batch size is 32.

#### Looking forward for 30% of LER

So, we saw that we split the test set and validation set different from Graves, and that using a batch size of 32 is better than 1. But, doing these modifications can we achieve the same results than him?

Running this simulation for over 750 epochs we got the following result:

![loss of best run](/assets/graves2006/loss_best.svg){:class="img-responsive img-center"}
![ler of best run](/assets/graves2006/ler_best.svg){:class="img-responsive img-center"}

We can notice that the bias between valid set and train set is getting worse, showing that the network wasn't able to generalize well, indicating that the use of random gaussian noise at the input of network wasn't a good regularizer. Getting the best model, that occurred at epoch 706, and testing with the test set, and applying the beam search decoder with width of 100 **we got a LER of** `29.64%`! This is a better result that was presented in the paper (remember that our training dataset is a bit bigger, and we decoded using the beam search)! :)

### Next steps

In a matter of fact, we haven’t exploited increasing the network depth yet. So in our next steps, we’ll try to apply the same dataset and cost function in deeper models, *i.e.*, with more layers.

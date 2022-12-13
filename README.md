# 6.7900_Project_Sensor

Here we are trying to training a regression model to predict the response of the DNA-Nanotube to specific analytes.
The data contains the sequence of DNA and pH of the solution.

Please see more in our article.

# Motication

Biosensors usually employ biomolecules such as enzymes, antibodies or nucleic acids to detect disease biomarkers. Nanotechnology has enabled the fabrication of novel biosensors with synthetic molecular recognition elements. More specifically, the use of DNA-wrapped SWNTs (single walled carbon nanotube) has been used for the detection of a wide range of analytes in biological media. Distinct changes in fluorescent intensity values from each DNA-SWNT combination were observed when the sensor successfully recognized the analyte of interest. 

In this project, we are interested to learn whether a specific DNA sequence or pH conditions would lead to stronger sensor response. We currently have experiment data of ~120 DNA-SWNT complexes’ response to the addition of serotonin. We formulate this as a classification problem, where we would use classification models to predict whether a DNA-SWNT complex is active. In detail, we would design a representation to encode our inputs and evaluate the distribution of the data. Then we would compare the performance of different classification models by our validation data-set. Finally, we would analyze why and how our models can conduct such results.
Our dataset is open-source.


# Data

Data are from experiments, containing:
> * DNA sequences
> * pH conditions
> * response to four analytes (chlor, cd, enro, semi)

we have 224 cases in total. for each case, we have at least `(39*embedding)+1+8` values involved.

We split it based on the ratio `training data : validation data = 8 : 2` to the training dataset (180) and validation dataset (44). 

## DNA embedding

We consider the DNA sequence as a language system, i.e. a sequence is a sentence under our envision. As a sequential data, the short and long range correlation of the DNA sequence may affect the property of the materials. For instance, some particular pattern in seq will form sepecific secondary sturcutre domain in DNA, thus leads to different results of experiments. 

To capture the motif of sequences of DNA, we intended to employ advance models, like CNN models. However, our data size (especially the labelled data) is limited by the throughout of the experiments. Thus, we formulate our situation as "few labelled target data" which was introduced in the course. A simple way to bypass the problem is to use a pre-trained model, which has leaned the DNA motif from tons of unlabelled data. 

Here, we emploied the work of Gonzalo Benegas, et al. The model, GPN (Genomic Pre-trained Network), was trained on the non-coding DNA dataset unsupervisedly. A DNA embedding matrix for DNA sequences will be produced by this model finally, and this embedding will be the input of our model, which has the shape `[seq, 512]`

> Gonzalo Benegas, Sanjit Singh Batra, Yun S. Song bioRxiv 2022.08.22.504706; doi: https://doi.org/10.1101/2022.08.22.504706

## pH embedding

In the experiments, we only had two pH conditions, pH = 6 / 8. We used a one-hot-like style to encode this. To align with the DNA embedding, we assigned the vector `[0, 0, ..., 0]` to pH = 6, and the vector `[1, 1, ..., 1]` to pH = 8. Each vector has the shape `[1, 512]`.

## Response & Outputs
The outputs of our model are 8-dim vectors (2 responses * 4 analytes).
Different responses have different distributions. To eliminate the effect of the imbalanced distribution, we normolized the first response by a scaling factor of 0.0113.

# Model

Our task is fomulated as a regression task, and our model is a simple MLP network.

## Inputs

We concatenate the two features, DNA embedding and pH embedding, as a whole. 

## Padding Seq

From the experiments, DNA owns different lengths. To align different long DNA, we padded all DNA seq to the longest one in them. The max number of length is 39, then all DNA embeddings will be padded with zero-matrix so that the final shape is `[39, 512]`.

With the embedding of pH, the shape of the inputs is `[40, 512]`

## Structure

The structure of our model is:
> `x -> Linear(512->8) -> ReLU -> Dropout(0.1) -> Linear(seq->1)`

the number of parameters is `520 + 8 + 41 + 1 = 570`. which is less than the number of the data points.


## Loss function

For a regression model, we used MSELoss as our loss fucntion, which has the formula:
$$
L = \frac{1}{n}\sum^n_1{(y-\hat{y})^2}
$$

## Regularization

For simplicity, we used `dropout` layer to regularize the models. The `dropout rate` is set to be `0.1`.

> TODO, can investigate the effect of the regularization.
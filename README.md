# RONN
Reinforced Optimizer for Neural Networks

Author/Lead: Kelly Odgers

## Goal:
Investigation into training an RL Agent to micromanage the learning process of a 
neural network.

## SubGoals:
  * Maintain a Keras Model Subclass which calculates and records the Loss Change Allocation
metric for the network.
    
  * Develop and maintain a custom optimizer which allows per layer configuration of
    learning rate and momentum
    
## Current Project Status:
  * Under development. 
    
  * Contact KOdgers if you would like to contribute or have any questions

## Test Cases:

### Cover_type Sklearn Data Set

### Higgs Tensorflow Data Set

## Requirements

- Python 3.8
- Tensorflow 2.4
- Scikit-learn
- TF-Agents
- Tensorflow-datasets
- Pandas
- psutils
- matplotlib
- keras-tuner

## Benchmarking

For each valid combination of network model and dataset, benchmark tests are 
performed where a "standard" optimizer is used. The choice of optimizer and 
starting learning rate is optimized by keras-tuner. After tuning, the configuration
is evaluated 5 times to estimate expected optimal performance with simple learning 
(no lr scheduler).
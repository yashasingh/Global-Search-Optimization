# Global-Search-Optimization

### UNDERSTANDING THE CODE

The following implementation can be divided into three modules-
1. Firefly structure, definition and algorithm implementation.

2. Designing a modular deep learning model  which can be triggered dynamically by firefly algorithm.

3. Implementation of parallel processing techniques for performance optimization. 

For deep model optimization, we choose learning rate (referred by variable x), number of hidden layers (referred by variable y) and number of nodes in each layer (referred by variable z) as tuning hyperparameters. This results in a dynamic structure of fireflies since z is dependent on the number of hidden units.

All the designed models used Gradient Descent algorithm as the optimizer for weight optimization. Each layer used sigmoid function as the activation function and  softmax cross-entropy in last layer for classifying labels.



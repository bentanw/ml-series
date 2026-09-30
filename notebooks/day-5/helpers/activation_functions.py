import numpy as np


def sigmoid(z):
    """
    Sigmoid converts the score into a probability:

    score -5 → 0.007 → probably not a cat
    score  0 → 0.500 → uncertain
    score  5 → 0.993 → probably a cat

    Purpose:
    - useful for answering one yes-or-no question.
    """
    return 1 / (1 + np.exp(-z))


def relu(z):
    """
    A neuron examines all the pixels and calculates a score:

    score is negative → “I did not detect my pattern.”
    score is positive → “I detected my pattern this strongly.”

    ReLU handles that score like this:

    negative score → output 0
    positive score → keep the score

    For example:
    Neuron's score: -4  → ReLU → 0
    Neuron's score: -1  → ReLU → 0
    Neuron's score:  2  → ReLU → 2
    Neuron's score:  7  → ReLU → 7

    Purpose:
    - It ignores that particular neuron’s contribution for that example.
    - pretty much saying: “I didn’t detect the pattern, so I have nothing useful to report.”
    """
    return np.maximum(0, z)


def softmax(z):
    """
    The output neurons initially produce arbitrary numbers called logits:

    logits = [1.2, 3.7, -0.5, 0.8, 0.1]

    Softmax converts them into probabilities:

    probabilities = [0.05, 0.87, 0.01, 0.04, 0.03]

    - Are all between 0 and 1
    - Add up to 1
    - Compete with one another

    Purpose:
    - The probabilities represent the likelihood that each category is the correct answer.
    - useful for choosing one answer from multiple classes.
    """
    # Normalize over classes for each example; shifting prevents overflow.
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

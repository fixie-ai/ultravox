# Evaluations for EPD

## Base

{'accuracy': 0.5769230769230769}
Confusion matrix:
[[0.66666667 0.33333333]
 [0.54545455 0.45454545]]

## 3x END

{'accuracy': 0.46153846153846156}
Confusion matrix:
[[0.13333333 0.86666667]
 [0.09090909 0.90909091]]

## 2 END 8 Random ... (shift left)

{'accuracy': 0.5384615384615384}
Confusion matrix:
[[0.2 0.8]
 [0.  1.]]

## Same but shift right

{'accuracy': 0.7307692307692307}
Confusion matrix:
[[0.6        0.4       ]
 [0.09090909 0.90909091]]

## 8->16 random ... (reverted)

{'accuracy': 0.6923076923076923}
Confusion matrix:
[[0.66666667 0.33333333]
 [0.27272727 0.72727273]]

## 30% crop (not trained fully)

{'accuracy': 0.6153846153846154}
Confusion matrix:
[[0.33333333 0.66666667]
 [0.         1.]]

## + 40% silence

{'accuracy': 0.5769230769230769}
Confusion matrix:
[[1. 0.]
 [1. 0.]]

## + system prompt

{'accuracy': 0.6923076923076923}
Confusion matrix:
[[0.8        0.2       ]
 [0.45454545 0.54545455]]

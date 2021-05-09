# ASK: Adversarial Soft KNN

## CIFAR10 ASK robust training
`python ask_train.py --dataset=cifar10`

## Imagenette ASK robust training
`python ask_train.py --dataset=imagenette --batch-size=64 --lr=0.01 --eps-train=2 --eps-eval=4 --eps-ask=4 --step-size=1 --dknn-size=8500 --hidden-layer=2`

Note 1: please add `d` flag the first time you run a model on a dataset

Note 2: please use the `disable-ask` option to run standard adversarial training

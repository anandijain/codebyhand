# codebyhand

i want to code with a pen tablet, but there are no good open source models that merge with popular IDEs like vscode to integrate with intellij

## What's in the repo

* python paint program to save handwritten characters to disk(english only rn)
* file to train basic net on the EMNIST dataset
* file (kinda) to eval saved digits based on trained model (basic interpolation for resizing)
* it gets basic single stroke characters accurately sometimes, but i need to mess with the transformers

## TODO/GOAL more feasible

* save images for new labeled data given by user, ideally save pixel path too
* decently accurate when evaluating on my own handwriting
* multi-stroke character support: using detectron2 to segment characters, then place on white background before eval
* save strokes to database

## TODO/GOAL less likely/feasible

* distributed training
* possibly running on a flask app
* idek about integrating with vscode, but I guess i could probably deploy an ONYX model in the extension

## COMPLETED

* integrated trained emnist model with paint.py so that as strokes are written, the model live infers on them and prints to stdout
* train model as you write (giving target chars)

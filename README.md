# codebyhand

i want to code with a pen tablet, but there are no good open source models that merge with popular IDEs like vscode to integrate with intellij

## What's in the repo

* python paint program to save handwritten characters to disk(english only rn)
* file to train basic net on the EMNIST dataset (which torchvision just fixed and needs to be merged)
* file (kinda) to eval saved digits based on trained model (basic interpolation for resizing)

## TODO/GOAL

* decently accurate when evaluating on my own handwriting
* integrating with paint.py so that as strokes are written, the model evals on them and prints to stdout
* possibly running on a flask app
* idek about integrating with vscode, but I guess i could probably deploy an ONYX model in the extension
# usbtablet


i want to code with a pen tablet, but there are no good open source models

linx:

credit for base paint program
https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

to train 
https://www.nist.gov/itl/products-and-services/emnist-dataset # a-z A-Z 0-9


to make it actually work:
    - train conv net on whatevers out there
    - bounding box detection for characters to rescale to pretrained.
    - probably want to include an nlp model that predicts the next word


use cases:
    - vscode utility to easily match handwriting predictions with intellij


gtrans handwriting tool spec:

    ~10 suggestions, punctuation to start
    ~ backspace
    ~ space
    ~ enter
    ~ unique icon when hovering/writing
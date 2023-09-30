
import random

# placeholder picker for model index
# this function can be customized to pick
# smarter the model for the treenode each time
# maybe using history (statistics) of models with
# that index in previous nodes in the tree

def model_picker(models_list):
    leng = len(models_list)
    ind = random.randint(0,leng-1)
    return models_list[ind]

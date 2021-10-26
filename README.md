# Few-shot-paddle

The aim for this repository is to contain clean, readable and tested code to reproduce few-shot learning research. See [pytorch implements here](https://github.com/oscarknagg/few-shot).

This project is written in python3.6+ and paddle in AI Studio and assumes you have a GPU.

See these Medium articles for some more information

- [Theory and concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
- [Discussion of implementation details](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)

# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r requirements.txt` preferably in a virtualenv.

### Data
Edit the `DATA_PATH` variable in `config.py` to the location where you store the Omniglot and miniImagenet datasets.

After acquiring the data and running the setup scripts your folder structure should look like
```
DATA_PATH/
    Omniglot/
        images_background/
        images_evaluation/
    miniImageNet/
        images_background/
        images_evaluation/
```

**Omniglot** dataset. Download from https://github.com/brendenlake/omniglot/tree/master/python, place the extracted files into `DATA_PATH/Omniglot_Raw` and run `prepare_omniglot.py`

## Matching Networks

**Arguments**

- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n-train: Support samples per class for training tasks
- n-test: Support samples per class for validation tasks
- k-train: Number of classes in training tasks
- k-test: Number of classes in validation tasks
- q-train: Query samples per class for training tasks
- q-test: Query samples per class for validation tasks
- fce: Whether (True) or not (False) to use full context embeddings (FCE)
- lstm-layers: Number of LSTM layers to use in the support set
    FCE
- unrolling-steps: Number of unrolling steps to use when calculating FCE
    of the query sample

I had trouble reproducing the results of this paper using the cosine distance metric as I found the converge to be slow and final performance dependent on the random initialisation. However I was able to reproduce (and slightly exceed) the results of this paper using the l2 distance metric.


|                     | Omniglot|
|---------------------|---------|
| **k-way**           | **5**   |
| **n-shot**          | **1**   |
| Pytorch Published (l2) | 98.3   |
| This paddle Repo (l2) | 98.85   |

 You can run it like below.
 ```
 python -m matching_nets.py --dataset omniglot --fce False --k-test 5 --n-test 1 --distance l2
 ```

**Final**

- See pretrained model in **pretrained** file.

- You can see each step of our implements with paddle compares to pytorch in **pipeline**.


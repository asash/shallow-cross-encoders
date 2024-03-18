Code for the ECIR 2024 paper "Shallow Cross-Encoders for Low-Latency Retrieval". 

## Credits: 
[Aleksandr V. Petrov](https://asash.github.io) <BR/>
[Sean MacAvaney](https://macavaney.us/) <BR/>
[Craig Macdonald](https://www.dcs.gla.ac.uk/~craigm/)

**If you use this code, please consider citing the paper:**: 

```
@article{petrov2023shallow,
  title={Shallow Cross-Encoders for Low-Latency Retrieval},
  author={Petrov, Aleksandr and Macdonald, Craig and MacAvaney, Sean},
  @booktitle={European Conference on Information Retrieval}
  year={2024}
}
```

# Instructions 
## Dependencies
To run the code, please install the following dependencies: [pytorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index), [pyterrier](https://pyterrier.readthedocs.io/en/latest/installation.html), [pyterrier-pisa](https://github.com/terrierteam/pyterrier_pisa), [ir-datasets](https://ir-datasets.com/), [ir-measures](https://ir-measur.es/en/latest/)


# Runnign the code
To train a shallow cross-encoder on the MS-Marco dataset, run 

```
python3 train_shallow_crossencoder.py
```

Some useful parameters:
| --backbone-model | backbone model; 'prajjwal1/bert-tiny, 'prajjwal1/bert-mini' or 'prajjwal1/bert-small'  |
|------------------|----------------------------------------------------------------------------------------|
| -t               | Parameter t for the gBCE loss; we recommend to set it to 0.75                          |
| --negs           | Number of negatives per positive; defaults to 16                                       |


To compare with the baselines run python3 `evaluate_tinybert.py`. When evaluating, make sure that you've replaced the model checkpoints specified in the evaluation code. 


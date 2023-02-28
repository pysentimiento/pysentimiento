
# Training models

To train a model, use the script `bin/train.py`, which takes the following arguments:


```bash
python bin/train.py --base_model <model_name> \
    --task <task> \
    --lang <lang> \
    --push_to <huggingface_identifier>
```

to train and push a model to HuggingFace. For instance, to train a RoBERTuito model for Sentiment Analysis, use:

```bash
python bin/train.py --base_model "pysentimiento/robertuito-base-uncased"\
    --push_to 'pysentimiento/robertuito-sentiment-analysis'\
    --task sentiment --lang es
```

Note that if you don't belong to [pysentimiento](https://huggingface.co/pysentimiento) organization in huggingface, this won't work because you don't have access to data.


## Benchmarking

To run benchmarks you can use also `bin/train.py` passing the `--benchmark`

```bash

python bin/train.py --base_model "pysentimiento/robertuito-base-uncased"\
    --task sentiment --lang es --benchmark --times 10
```

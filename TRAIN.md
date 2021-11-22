
# Training models

To train a model, use the script `bin/train.py`, which takes the following arguments:


```bash
python bin/train.py <model_name> \
    --output_path <output_path> \
    --task <task> \
    --epochs 5 \
    --lang <lang>
```

For instance, to train a BETO model for Sentiment Analysis, use:

```bash
python bin/train.py "dccuchile/bert-base-spanish-wwm-uncased"\
    --output_path models/beto-sentiment-analysis/ \
    --task sentiment --lang es
```


To train the current best-performing models for each language, run:

```bash
./train_classifiers_es.sh
```




## Benchmarking

To run benchmarks you can use also `bin/train.py` passing the `--benchmark`

```bash
python bin/train.py "dccuchile/bert-base-spanish-wwm-uncased" --lang es --benchmark --benchmark_output_path benchmark_path.json
```

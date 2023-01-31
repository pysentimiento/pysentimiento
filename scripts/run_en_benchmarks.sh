#/bin/bash
models=(
    "bert-base-uncased"
    "roberta-base"
    "vinai/bertweet-base"
    "pysentimiento/robertuito-base-uncased"
    "google/electra-base-discriminator"
)


tasks = (
    "sentiment"
    "emotion"
    "irony"
    "hate_speech"
)

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        python bin/train.py --base_model $model \
            --lang es \
            --task $task \
            --benchmark --times 10 \
    done
done

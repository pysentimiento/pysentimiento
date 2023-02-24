#/bin/bash
models=(
    "bert-base-uncased"
    "roberta-base"
    #"vinai/bertweet-base"
    "pysentimiento/robertuito-base-uncased"
    "google/electra-base-discriminator"
)


tasks=(
    "irony"
    #"sentiment"
    #"emotion"
    #"hate_speech"
)

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        echo "Running benchmark for $model and $task"
        python bin/train.py --base_model $model \
            --lang en \
            --task $task \
            --benchmark --times 10
    done
done
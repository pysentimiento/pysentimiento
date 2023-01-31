#/bin/bash
models=(
    "bert-base-uncased"
    "roberta-base"
    "vinai/bertweet-base"
    "pysentimiento/robertuito-base-uncased"
    "google/electra-base-discriminator"
)

tasks=(
    "sentiment"
    "emotion"
    #"irony"
    #"hate_speech"
)

lang="en"

for model in "${models[@]}"
do

    for task in "${tasks[@]}"
    do
        echo "Running hyperparameter tuning for $model and $task"
        # Run hyperparameter tuning
        python bin/hp_tune.py --model $model \
            --lang $lang \
            --task $task \
            --count 40
    done
done

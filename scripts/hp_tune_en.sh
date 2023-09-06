#/bin/bash

if [ -z "$1" ]
then
    tasks=(
        "irony"
        "sentiment"
        "emotion"
        "hate_speech"
    )
else
    tasks=("$1")
fi


# if model is empty, run all models

if [ -z "$2" ]
then
    models=(
        "bert-base-uncased"
        "roberta-base"
        "vinai/bertweet-base"
        "pysentimiento/robertuito-base-uncased"
        "google/electra-base-discriminator"
    )
else
    models=("$2")
fi

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

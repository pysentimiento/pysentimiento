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
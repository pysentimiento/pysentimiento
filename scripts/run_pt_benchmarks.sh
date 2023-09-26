#/bin/bash

# If tasks is empty, run all tasks
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
        "melll-uff/bertweetbr"
        "neuralmind/bert-base-portuguese-cased"
        "pablocosta/bertabaporu-base-uncased"
        "pysentimiento/robertuito-base-uncased"
        #"rdenadai/BR_BERTo" Not a good model
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
            --lang pt \
            --task $task \
            --benchmark --times 10
    done
done
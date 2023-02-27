#/bin/bash
models=(
    #"melll-uff/bertweetbr"
    "neuralmind/bert-base-portuguese-cased"
    "pablocosta/bertabaporu-base-uncased"
    "pysentimiento/robertuito-base-uncased"
    #"rdenadai/BR_BERTo"
)

tasks=(
    #"irony"
    "sentiment"
    #"emotion"
    "hate_speech"
)

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
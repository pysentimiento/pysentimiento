#/bin/bash
models=(
    "melll-uff/bertweetbr"
    "neuralmind/bert-base-portuguese-cased"
    "pablocosta/bertabaporu-base-uncased"
    "pysentimiento/robertuito-base-uncased"
    #"rdenadai/BR_BERTo"
)

tasks=(
    #"sentiment"
    "emotion"
    #"irony"
    #"hate_speech"
)

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        echo "Running hyperparameter tuning for $model and $task (pt)"
        # Run hyperparameter tuning
        python bin/hp_tune.py --model $model \
            --lang pt \
            --task $task \
            --count 25
    done
done

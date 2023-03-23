#/bin/bash
models=(
    "dccuchile/bert-base-spanish-wwm-cased"
    "PlanTL-GOB-ES/roberta-base-bne"
    "bertin-project/bertin-roberta-base-spanish"
    "pysentimiento/robertuito-base-uncased"
    "mrm8488/electricidad-base-discriminator"
)

tasks=(
    #"sentiment"
    #"emotion"
    #"irony"
    #"hate_speech"
    "targeted_sa"
)

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        echo "Running benchmark for $model and $task"
        python bin/train.py --base_model $model \
            --lang es \
            --task $task \
            --benchmark --times 10
    done
done

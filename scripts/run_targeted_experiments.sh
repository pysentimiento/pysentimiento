#/bin/bash
models=(
    "pysentimiento/robertuito-base-uncased"
    "bertin-project/bertin-roberta-base-spanish"
    "mrm8488/electricidad-base-discriminator"
    "dccuchile/bert-base-spanish-wwm-cased"
    "PlanTL-GOB-ES/roberta-base-bne"
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
            --benchmark --times 10 --untargeted

        python bin/train.py --base_model $model \
            --lang es \
            --task $task \
            --benchmark --times 10
        rm -Rf tmp/*
        rm -Rf results/*
    done
done

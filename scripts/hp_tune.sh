#/bin/bash
models=(
    "dccuchile/bert-base-spanish-wwm-cased"
    "PlanTL-GOB-ES/roberta-base-bne"
    #"bertin-project/bertin-roberta-base-spanish"
    #"pysentimiento/robertuito-base-uncased"
    #"mrm8488/electricidad-base-discriminator"
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
        echo "Running hyperparameter tuning for $model and $task"
        # Run hyperparameter tuning
        python bin/hp_tune.py --model $model \
            --lang es \
            --task $task \
            --count 40
    done
done

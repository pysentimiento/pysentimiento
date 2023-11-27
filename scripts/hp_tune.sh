#/bin/bash

if [ -z "$1" ]
then
    tasks=(
        "irony"
        "sentiment"
        "emotion"
        "hate_speech"
        "targeted_sa"
        "context_hate_speech"
    )
else
    tasks=("$1")
fi


# if model is empty, run all models

if [ -z "$2" ]
then
    models=(
        "pysentimiento/robertuito-base-uncased"
        "dccuchile/bert-base-spanish-wwm-cased"
        "PlanTL-GOB-ES/roberta-base-bne"
        "bertin-project/bertin-roberta-base-spanish"
        "mrm8488/electricidad-base-discriminator"
    )
else
    models=("$2")
fi

for model in "${models[@]}"
do

    for task in "${tasks[@]}"
    do
        echo "Running hyperparameter tuning for $model and $task"
        # Run hyperparameter tuning
        python bin/hp_tune.py --model $model \
            --lang es \
            --task $task \
            --count 30
    done
done

output_path="evaluations/es/beto_cased.json"
python bin/train.py --base_model "dccuchile/bert-base-spanish-wwm-cased"\
    --lang es \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/beto_uncased.json"
python bin/train.py --base_model "dccuchile/bert-base-spanish-wwm-uncased"\
    --lang es \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/mbert_uncased.json"
python bin/train.py --base_model "bert-base-multilingual-uncased" \
    --lang es \
    --benchmark --benchmark_output_path $output_path \
    --overwrite



output_path="evaluations/es/roberta.json"
python bin/train.py --base_model "BSC-TeMU/roberta-base-bne" \
    --lang es \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/bertin.json"
python bin/train.py --base_model "bertin-project/bertin-roberta-base-spanish" \
    --lang es \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/robertuito.json"
python bin/train.py --base_model "pysentimiento/robertuito-base-uncased" \
    --lang es \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

rm -Rf lightning_logs
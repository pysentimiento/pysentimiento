output_path="evaluations/en/bert.json"
python bin/train.py --base_model "bert-base-uncased"\
    --lang en \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/en/roberta.json"
python bin/train.py --base_model "roberta-base"\
    --lang en \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/en/mbert_uncased.json"
python bin/train.py --base_model "bert-base-multilingual-uncased" \
    --lang en \
    --benchmark --benchmark_output_path $output_path \
    --overwrite


output_path="evaluations/en/bertweet.json"
python bin/train.py --base_model "vinai/bertweet-base" \
    --lang en \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

#
# Cross
#
output_path="evaluations/en/robertuito.json"
python bin/train.py --base_model "pysentimiento/robertuito-base-uncased" \
    --lang en \
    --benchmark --benchmark_output_path $output_path \
    --overwrite


output_path="evaluations/en/beto.json"
python bin/train.py --base_model "dccuchile/bert-base-spanish-wwm-uncased"\
    --lang en \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

rm -Rf lightning_logs
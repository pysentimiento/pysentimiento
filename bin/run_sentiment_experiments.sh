embeddings_path="embeddings/cc.es.300.bin"
task="sentiment"
model_name="rnn"

output_path="evaluations/${task}/es/rnn_cc.json"
python bin/train.py --base_model $model_name\
    --lang es --task ${task} \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/${task}/es/birnn_cc.json"
python bin/train.py --base_model $model_name\
    --lang es --task ${task} \
    --bidirectional \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite



embeddings_path="embeddings/tweet_dim_300_ws_5.bin"
output_path="evaluations/${task}/es/rnn_twitter.json"
python bin/train.py --base_model $model_name\
    --lang es --task ${task} \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/${task}/es/birnn_twitter.json"
python bin/train.py --base_model $model_name\
    --lang es --task ${task} \
    --bidirectional \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

rm lightning_logs -Rf
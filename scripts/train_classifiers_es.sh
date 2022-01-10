python bin/train.py pysentimiento/robertuito-base-uncased \
    --output_path models/robertuito-sentiment-analysis \
    --task sentiment \
    --epochs 5 --lang es

python bin/train.py pysentimiento/robertuito-base-uncased \
    --output_path models/robertuito-emotion-analysis \
    --task emotion \
    --epochs 5 --lang es


python bin/train.py pysentimiento/robertuito-base-uncased \
    --output_path models/robertuito-irony \
    --task irony \
    --epochs 5 --lang es

#
# Note: we train hate speech on task b, because it gives better results
# and its output is richer (contains HS, TR, AG, see hatEval paper for more info)
#
python bin/train.py pysentimiento/robertuito-base-uncased \
    --output_path models/robertuito-hate-speech \
    --task hate_speech \
    --epochs 5 --lang es --task_b


python bin/train.py --base_model pysentimiento/robertuito-base-deacc \
    --output_path models/robertuito-lince-ner \
    --lang es --epochs 10 --task ner

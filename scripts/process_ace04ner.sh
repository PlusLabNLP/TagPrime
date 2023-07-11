export ORIDATA_PATH="./ACE0405_NER"
export OUTPUT_PATH="./processed_data/ace04ner_bart"

mkdir $OUTPUT_PATH

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/train.ACE2004.jsonlines -o $OUTPUT_PATH/train.json -b facebook/bart-large 

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/dev.ACE2004.jsonlines -o $OUTPUT_PATH/dev.json -b facebook/bart-large  

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/test.ACE2004.jsonlines -o $OUTPUT_PATH/test.json -b facebook/bart-large 


export OUTPUT_PATH="./processed_data/ace04ner_t5"

mkdir $OUTPUT_PATH

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/train.ACE2004.jsonlines -o $OUTPUT_PATH/train.json -b t5-base 

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/dev.ACE2004.jsonlines -o $OUTPUT_PATH/dev.json -b t5-base 

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/test.ACE2004.jsonlines -o $OUTPUT_PATH/test.json -b t5-base 

export OUTPUT_PATH="./processed_data/ace04ner_roberta"

mkdir $OUTPUT_PATH

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/train.ACE2004.jsonlines -o $OUTPUT_PATH/train.json -b roberta-large 

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/dev.ACE2004.jsonlines -o $OUTPUT_PATH/dev.json -b roberta-large 

python preprocessing/process_ace0405ner.py -i $ORIDATA_PATH/test.ACE2004.jsonlines -o $OUTPUT_PATH/test.json -b roberta-large  
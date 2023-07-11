## Base Setting

LANG=en
export RAW_DATA_PATH="./MTOP/${LANG}"

export OUTPUT_PATH="./processed_data/mtop_${LANG}/"
mkdir $OUTPUT_PATH


for SPLIT in train dev test
do
    python preprocessing/process_tosp.py -i $RAW_DATA_PATH -o $OUTPUT_PATH/$SPLIT -s $SPLIT
done


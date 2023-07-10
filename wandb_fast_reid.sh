
source ~/venv/fastreid/bin/activate 

LONG_OUTPUT=$(python wandb_fast_reid.py "$@")
i=0

while true;
do
        ((i++))
        OUTPUT_DIR=$(echo $LONG_OUTPUT | cut -d ',' -f $i)

        if [[ "$OUTPUT_DIR" = " 'OUTPUT_DIR'" ]]; then
                ((i++))
                OUTPUT_DIR=$(echo $LONG_OUTPUT | cut -d ',' -f $i)
                break
        fi

        if [ $i -ge 10000 ]; then
            exit 1
        fi
done

python wandb_log_fast_reid.py "$OUTPUT_DIR"

exit 0

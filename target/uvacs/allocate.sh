#   a100-dgx:
#     gres: gpu:a100:1
#     reservation: bi_fox_dgx
#     partition: bii-gpu
#     account: bii_dsc_community

#!/bin/bash



export TIME="24:00:00"
# export TIME="12:00:00"
export CORES=8

if [[ $1 == "a100" ]]; then
    export GPU="a100"
    export TIME="12:00:00"
    export CORES=8
    /opt/rci/bin/ijob -n ${CORES} --gres=gpu:${GPU}:1 --partition=gpu --account=bii_dsc_community --time=${TIME}
elif [[ $1 == "dgx" ]]; then
    export GPU="a100"
    export RESERVATION="bi_fox_dgx"
    export PARTITION="bii-gpu"
    export ACCOUNT="bii_dsc_community"
    /opt/rci/bin/ijob -n ${CORES} --gres=gpu:"${GPU}":1 --partition=$PARTITION --reservation=$RESERVATION --account=$ACCOUNT --time=${TIME}

else
    echo "Invalid parameter. Please specify 'a100' or 'dgx'."
    exit 1
fi

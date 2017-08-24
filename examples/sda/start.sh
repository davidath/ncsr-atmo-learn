#! /bin/bash
count=`ls t10k* | wc -l`
if [ $count -eq 0 ]
then
./get_mnist.sh
fi
let END=$1-1
range=$(seq 0 $END)
for i in $range;do
    python autoencoder.py -i autoenc_$i.ini -t train | tee autoenc_$i.txt
done
python stackedautoencoders.py -i autoenc_$i.ini -n $1 -t train -p autoenc | tee sda.txt

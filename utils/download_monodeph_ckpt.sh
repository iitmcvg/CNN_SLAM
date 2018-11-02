# Downloads all checkpoints for resnet 50 variants

aria2c --file-allocation=none -c -x 16 -s 16 --max-concurrent-downloads=5 -d checkpoints --input-file utils/aria_ckpt_input_file.txt

cd checkpoints
for i in *.zip; do unzip "$i" -d "${i%%.zip}"; done
rm *.zip
cd ..

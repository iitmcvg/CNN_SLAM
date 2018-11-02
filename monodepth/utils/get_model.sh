model_name=$1
output_location=$2

filename=$model_name.zip

url=http://visual.cs.ucl.ac.uk/pubs/monoDepth/models/$filename

output_file=$output_location/$filename

echo "Downloading $model_name"
aria2c --file-allocation=none -c -x 16 -s 16 -d $output_location $url

#wget -nc $url -O $output_file

unzip $output_file -d $output_location
rm $output_file
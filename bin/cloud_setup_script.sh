#! /bin/bash
# GPU drivers
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda -y
nvidia-smi

# Set up conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH=$PATH:~/miniconda/bin
conda init bash
exec bash -l

# Set up repo and data
git clone https://github.com/jukiewiczm/kaggle-predict-future-sales.git
cd kaggle-predict-future-sales
git checkout implement-guild-tuning # to be removed later
rm -rf data/

# Download the data from my google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R6qX7LwdIwFdb0zw4yD1Tejk2YLoMpdO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R6qX7LwdIwFdb0zw4yD1Tejk2YLoMpdO" -O data.tar.gz && rm -rf /tmp/cookies.txt
tar -xf data.tar.gz
rm data.tar.gz
# And the wiki.ru.vec dict
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.vec
mv wiki.ru.vec data/

# Other requirements
sudo apt-get install cmake -y
sudo apt-get install g++ -y

# Set up conda
conda env create -f environment.yml --prefix ./env
sed -i "s/^ENV_PATH.*$/ENV_PATH=$(pwd | sed -e 's/[\/&]/\\&/g')\/env\//" ./bin/install_starspace.sh
bash ./bin/install_starspace.sh

# # After the init is done
# screen
# conda activate ./env
# # My local env has improper pytorch version for cloud GPU
# conda uninstall pytorch torchvision cudatoolkit -y
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
# # And for some reason this is needed for guild to work properly
# pip uninstall typing -y
# cd modeling/models/rnn
# guild run tune_rnn

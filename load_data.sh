### Check if a directory does not exist ###
if [ ! -d "data/" ] 
then
    mkdir data/ && cd data/
    wget https://aclanthology.org/anthology+abstracts.bib.gz
    gzip -d anthology+abstracts.bib.gz
    sleep 3

    curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3687{/ud-treebanks-v2.8.tgz,/ud-documentation-v2.8.tgz,/ud-tools-v2.8.tgz}
    tar -xzf ud-treebanks-v2.8.tgz

fi
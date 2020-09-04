# LessErrors
This repository contains a system designed to correct grammatical errors without using a parallel data.

More information about this system can be found [here](Thesis_yee0.pdf).

# Installation
```
python3 -m venv lesserrors_env
source lesserrors_env/bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en
wget https://github.com/nreimers/truecaser/releases/download/v1.0/english_distributions.obj.zip
unzip english_distributions.obj.zip -d truecaser && rm english_distributions.obj.zip
```

### KenLM
KenLM is a language modelling toolkit, which is used to assess grammaticallity in our system.
Here we installed KenLM as follows:

Navigate to the directory you want to install KenLM and then run:  
```
git clone https://github.com/kpu/kenlm.git
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j 4
pip3 install https://github.com/kpu/kenlm/archive/master.zip
```
This installs and compiles the main program along with a python wrapper.  

Having installed KenLM, we then need to build a language model from native text. We used the [Billion Word Benchmark](http://www.statmt.org/lm-benchmark/) dataset, consisting of close to a billion words of English taken from news articles on the web. The text can be downloaded and preprocessed as follows:  
```
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -zxvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* > 1b.txt
```
This will produce a 3.86GB text file called 1b.txt which contains the entire training set, one sentence per line.

To train a model on this file with KenLM, run:
```
mkdir tmp
/<your_path>/kenlm/build/bin/lmplz -o 5 -S 50% -T tmp/ < 1b.txt > 1b.arpa
```
-o lets you choose n-gram order (we recommend 5)  
-S lets you specify how much RAM to use in percent (can also say e.g. 10G for 10GB)  
-T lets you choose a directory for temporary files (recommended)  
This may take some time depending on your hardware. On our system, the model took ~90mins to train.

Having trained a model, finally run the following to binarise it.  
`/<your_path>/kenlm/build/bin/build_binary 1b.arpa /<lesserrors_path>/lmgec/resources/1b.bin` 
This will both reduce model loading times and save hard drive space.

# Run server
```
python3 manage.py runserver 0.0.0.0:<PORT_NUMBER>
```

# API examples
```python
import requests
def correct(sent, threshold=0.98, threshold_insert=0.92, generate_insertion_candidates=True, use_nli=True, use_truecase=True):
    url = "http://localhost:<PORT_NUMBER>/correct"
    r = requests.post(url, json={"sent": sent, "threshold": threshold, "threshold_insert": threshold_insert, "generate_insertion_candidates": generate_insertion_candidates, "use_nli": use_nli, "use_truecase": use_truecase})
    if r.status_code == requests.codes.ok:
        return r.text
    else:
        return r.text

sent = "I am looking forway see you."
print(correct(sent)) # 'I am looking forward to seeing you .'
```

### For [frontend](https://github.com/NTHU-NLPLAB/lesserrors-frontend) usage

```python
import requests
def correct_frontend(sent, threshold=0.98, threshold_insert=0.92, generate_insertion_candidates=True, use_nli=True, use_truecase=True):
    url = "http://localhost:<PORT_NUMBER>/lesserrors"
    r = requests.post(url, json={"sent": sent, "threshold": threshold, "threshold_insert": threshold_insert, "generate_insertion_candidates": generate_insertion_candidates, "use_nli": use_nli, "use_truecase": use_truecase})
    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        return r.text

sent = "I am looking forway see you."
print(correct_frontend(sent))
# {'proc': [[0, ['I am looking forway see you .', -3.45, '--', '--']], [1, ['I am looking [-forway-] {+forward+} see you .', -2.44, 1.01, 'SPELL']], [2, ['I am looking forward {+to+} see you .', -1.55, 0.89, 'INSERT']], [3, ['I am looking forward to [-see-] {+seeing+} you .', -1.21, 0.34, 'MORPH']]], 'result': [3, 'I am looking forward to seeing you .', 2.24]}
```

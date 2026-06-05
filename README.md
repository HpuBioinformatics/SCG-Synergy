# Install the Environment

## OS Requirements
The package development version is tested on *Linux* (Ubuntu 20.04) operating systems with CUDA 12.4.

## Python Dependencies
SCG-Synergy is tested under `python --<3.9.23>`

## Prerequisites


We provide a txt file containing the necessary packages for SCG-Synergy. All the required basic packages can be installed using the following command:
```
pip install -r requirements.txt
```

## Usage
To train and evaluate the model, you could run the following command.

**O'Neil Dataset**
```bash
python main.py --dataset 'ONEIL' --threshold 30 --alpha 0.2  --learning_rate 1e-4 --weight_decay 1e-3 --temperature 0.5
```

**NCI-ALMANAC Dataset**
```bash
  python main.py --dataset 'ALMANAC' --threshold 10 --alpha 0.05 --learning_rate 1e-4 --weight_decay 1e-3 --temperature 0.5
```
## Please enter the address to save the result
```bash
  OUTPUT_DIR =
```



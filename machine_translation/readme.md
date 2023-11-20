## Train text-to-text model for tranlation using NLLB datasets

* [Model training](https://github.com/eistakovskii/LR_Transfer/blob/patch-1/machine_translation/train_mt.py) - this python file carries out training of a given model for the task of translation.
* [Model evaluation](https://github.com/eistakovskii/LR_Transfer/blob/patch-1/machine_translation/evaluate_mt.py) - this python file intended to carry out a zero-shot evaluation on a given base model, checkpoint and an NLLB valid language pair. The python file also allows for a finetuning a base model if fullfineting option was not indicated at the beginning of a training.

The file expects to be given a [valid NLLB-type language pair](https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py) to download dataset and start working. First, check that your languages are present in the NLLB_PAIRS dictionary. If the required pair is present, pass the model your language string in the following format, e.g. pol_Latn-ukr_Cyrl

## How to run training

**Step $0$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $1$: Run Training**

``` shell
python train_mt.py --lang_pair pol_Latn-ukr_Cyrl \
--short_run 0 \
--epoch_num 10 \
--model_name google/mt5-small
```
## How to run evaluation

**Step $2$: Run Evaluation**

``` shell
python evaluate_mt.py --lang_pair pol_Latn-ukr_Cyrl \ # specify your language pair, required
--base_model_name google/mt5-small \ # specify your base model as a hg model path, optional
--do_finetune 0 \ # specify whether you want to finetune the base model
--checkpoint checkpoints/mt5_model.pth \ # specify the path to your checkpoint if you do not want to finetune the base model
--nickname test_run \ # give a nickname to your experiment
--short_run 1 \ # specify whether you want to quickly run a test on a small dataset or use fullsize dataset
```

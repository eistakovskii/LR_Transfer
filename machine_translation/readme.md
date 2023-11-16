## Train text-to-text model for tranlation using NLLB datasets

* [Model training](https://github.com/eistakovskii/LR_Transfer/blob/patch-1/machine_translation/train_mt.py) - this python file carries out training of a given model for the task of translation.

The file expects to be given a [valid NLLB-type language pair](https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py) to download dataset and start working. First, check that your languages are present in the NLLB_PAIRS dictionary. If the required pair is present, pass the model your language string in the following format, e.g. pol_Latn-ukr_Cyrl

## How to run training

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

``` shell
python train_mt.py --lang_pair pol_Latn-ukr_Cyrl \
--short_run 0 \
--epoch_num 10 \
--model_name google/mt5-small
```

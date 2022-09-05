from datasets import load_metric

metric = load_metric('glue', 'stsb')
model_checkpoint = 'microsoft/deberta-v3-small'

if __name__ == '__main__':
    print(metric)

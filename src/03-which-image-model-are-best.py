import pandas as pd




anPath = "/Users/work/_ai/projects/course22/pytorch-image-models/results/pytorch-image-models/results/"
df_results = pd.read_csv(f'{anPath}results-imagenet.csv')

def get_data(part, col):
    #model,top1,top1_err,top5,top5_err,param_count,img_size,crop_pct,interpolation
    #model,infer_samples_per_sec,infer_step_time,infer_batch_size,infer_img_size,param_count
    df = pd.read_csv(f'{anPath}benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').merge(df_results, on='model')
    df['secs'] = 1. / df[col]
    df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
    df = df[~df.model.str.endswith('gn')]
    df.loc[df.model.str.contains('in22'),'family'] = df.loc[df.model.str.contains('in22'),'family'] + '_in22'
    df.loc[df.model.str.contains('resnet.*d'),'family'] = df.loc[df.model.str.contains('resnet.*d'),'family'] + 'd'
    return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg')]


df = get_data('infer', 'infer_samples_per_sec')
print(df)
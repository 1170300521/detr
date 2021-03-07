import json
import fire
import os.path as osp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


def save_visualize(outputs, visualize_dir):
    """
    Save self-attention and cross-modal attention weights to files along with
    imgs and word queries

    outputs: model outputs dict
    visualize_dir: path to save files
    """
    if len(outputs['ids']) == 0:
        return
#    for k in outputs.keys():
#        if k not in ['sents', 'size']:
#            outputs[k] = outputs[k].cpu().detach().tolist()
    filename = osp.join(visualize_dir, str(outputs['ids'][0])+'.json')
    query_ids = outputs['pred_logits'][:,:,0]
    _, query_ids = query_ids.max(1)    
    att_dict = {
        'sents': outputs['sents'],
        'size': outputs['size'],
        'img': outputs['img'],
        'self_att': outputs['self_att'],
        'cross_att': outputs['cross_att'],
        'query_ids': query_ids.cpu().detach().tolist(),
    }
    with open(filename, 'w') as f:
        json.dump(att_dict, f)


def show(filename, split_sent=9):
    """
    Show self-attention and cross-attention
    Parameters:
        filename: filename of attention maps
        split_sent: the number of first N words to show
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    file_prefix = filename.split(".")[0]
    # get sentence words
    sents = [s.split() for s in data['sents']]
    self_att = np.array(data['self_att'])  # B x L x N x N
    cross_att = np.array(data['cross_att'])  # B x L x N x (H*W)
    b, l, n, _ = cross_att.shape
    # better visualization with seaborn
    cross_att = cross_att.reshape((b, l, n, 20, 20))[:, :, :, ::-1]
    subject = [(i,s[i]) for s, i in zip(sents, data['query_ids'])]
    for i in range(b):
        title = "Idx: " + str(subject[i][0]) + "; Target: "+ subject[i][1] + "; " + data['sents'][i]
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[25, 15])
        fig.suptitle(title, fontsize=30)
        for j in range(l):
            sns.heatmap(self_att[i][j][:split_sent, :split_sent], ax=axes[int(j/3)][j%3], 
                        annot=self_att[i][j][:split_sent, :split_sent],cmap='YlGn')
        fig.savefig(file_prefix+"_selfattn_"+str(i)+".png")
        plt.close(fig)
    print("Complete show " + filename)

if __name__ == "__main__":
    fire.Fire(show)

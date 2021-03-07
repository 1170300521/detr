import json
import os.path as osp

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
    att_dict = {
        'sents': outputs['sents'],
        'size': outputs['size'],
        'img': outputs['img'],
        'self_att': outputs['self_att'],
        'cross_att': outputs['cross_att'],
    }
    with open(filename, 'w') as f:
        json.dump(att_dict, f)



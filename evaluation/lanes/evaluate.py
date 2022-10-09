import os, tqdm, torch, json
import torch.distributed as dist
import numpy as np
import scipy.special

from evaluation.lanes.laneEval import LaneEval
from data_loader.data_loaders import LanesDataLoader

import wandb

def eval_lanes(config, model):
    """
    config: set of hyperparameters
    model: model to evaluate
    Evaluates lane detection accuracy and writes results to table
    """

    model.eval()
    save_dir = os.path.join(config._save_dir, 'evaluation')
    exp_name = 'tusimple_eval'
    exp_name = os.path.join(save_dir, exp_name)

    config['tusimple_dataloader']['args']['batch_size'] = 1
    config['tusimple_dataloader']['args']['shuffle'] = False

    run_test_tusimple(model, exp_name, config['arch']['griding_num'], False, config)
    synchronize()  # wait for all results
    if is_main_process():
        combine_tusimple_test(exp_name)
        res = LaneEval.bench_one_submit(os.path.join(exp_name + '.txt'),
                                        os.path.join(config['tusimple_dataloader']['args']['data_dir'], 'test_label.json'))
        res = json.loads(res)

        print('Creating Table')
        table2 = wandb.Table(columns=["Name", "Value"])
        print('Filling Table')
        for r in res:
            table2.add_data(str(r['name']), str(r['value']))
            dist_print(r['name'], r['value'])
        wandb.log({"examples": table2})
        print('Table filled')
    synchronize()

# Main test function ------------------------------------------------------------------------------------------------- #
def run_test_tusimple(net,exp_name,griding_num,use_aux,config, batch_size = 8):
    """
    net: model to be evaluated
    exp_name: name of experiment, used to store temporary results
    griding_name: number of column grids
    use_aux: bool, indicating whether auxiliary segmentation branch is used or not
    config: hyperparameters of model
    batch_size: batch size for running evaluatin
    runs evaluation and writes results to exp_name file
    """
    output_path = os.path.join(exp_name+'.%d.txt'% get_rank())
    fp = open(output_path,'w')
    # loader = get_test_loader(batch_size,data_root,'Tusimple', distributed)
    loader = LanesDataLoader(config, 'test')
    for i,data in enumerate(dist_tqdm(loader)):
        imgs,names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)[1]
        if len(out) == 2 and use_aux:
            out = out[0]
        for i,name in enumerate(names):
            tmp_dict = {}
            tmp_dict['lanes'] = generate_tusimple_lines(out[i],imgs[0,0].shape,griding_num)
            tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
             270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
             430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
             590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            tmp_dict['raw_file'] = name
            tmp_dict['run_time'] = 10
            json_str = json.dumps(tmp_dict)

            fp.write(json_str+'\n')
    fp.close()


# Helper function ---------------------------------------------------------------------------------------------------- #
"""
Helper functions of official UFLD implementation
"""

def is_main_process():
    return get_rank() == 0

def dist_tqdm(obj, *args, **kwargs):
    if can_log():
        return tqdm.tqdm(obj, *args, **kwargs)
    else:
        return obj

def can_log():
    return is_main_process()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def dist_print(*args, **kwargs):
    if can_log():
        print(*args, **kwargs)

def can_log():
    return is_main_process()

def combine_tusimple_test(exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(exp_name+'.%d.txt'% i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(exp_name+'.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)

def generate_tusimple_lines(out,shape,griding_num,localization_type='rel'):

    out = out.data.cpu().numpy()
    out_loc = np.argmax(out,axis=0)

    if localization_type == 'rel':
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(griding_num)
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)

        loc[out_loc == griding_num] = griding_num
        out_loc = loc
    lanes = []
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:,i]
        lane = [int(round((loc + 0.5) * 1280.0 / (griding_num - 1))) if loc != griding_num else -2 for loc in out_i]
        lanes.append(lane)
    return lanes
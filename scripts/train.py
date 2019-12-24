from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
import itertools
from argparse import ArgumentParser
import datetime
import socket
from collections import defaultdict

import numpy as np
from torch_geometric.data import Data, DataLoader
import torch
import tensorboardX
import yaml
from tqdm import tqdm

from datasets.load_synthetic import load_synthetic
from datasets.load_noaa import load_noaa
from datasets.load_traffic import load_traffic
from datasets.load_varicoef import load_varicoef
from datasets.load_sst import load_sst
from models.PDGN import PDGN
from models.GraphPDE import GraphPDE
from models.baselines import JointVAR, JointMLP, JointRNN, SingleVAR, SingleMLP, SingleRNN
from models.utils import unbatch_node_feature_mat, unbatch_node_feature_mat_tonumpy, DataLoaderWithPad
from models.linear_reg_op import LinearRegOp
from utils.train_utils import MyArgs, save_ckpt, load_ckpt, get_last_ckpt, print_2way
from utils.train_utils import AlwaysSampleScheduler, InverseSigmoidDecaySampleScheduler


def main():
    parser = ArgumentParser()
    # train/test hyper parameters
    parser.add_argument('--dataset', type=str, default='noaa')
    parser.add_argument('--noaa_target_features', type=str, default='TEMP')
    parser.add_argument('--noaa_given_input_features', type=str, default='')
    parser.add_argument('--pde_mode', type=str, default='diff')
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--use_dist', type=str, default='False')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int ,default=1)
    parser.add_argument('--model', type=str, default='singlevar')
    parser.add_argument('--hidden_dim', type=int, default=4)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--pde_params_out_dim', type=int, default=None)
    parser.add_argument('--epochnum', type=int, default=5000)
    parser.add_argument('--validint', type=int, default=100)
    parser.add_argument('--testint', type=int, default=500)
    parser.add_argument('--ckptint', type=int, default=50)
    parser.add_argument('--sample_scheduler', type=str, default='always')
    parser.add_argument('--invsig_ss_itern', type=int, default=None,
                        help='hyper param for inverse sigmoid decay, '
                             'sample ratio drops to 0.5 after invsig_ss_itern / 2 epochs')
    parser.add_argument('--invsig_delay_start', type=int, default=0)
    parser.add_argument('--skip_first_frames_num', type=int, default=2)
    parser.add_argument('--return_features', type=str, default='False')
    parser.add_argument('--reinit_optimizer', type=str, default='False')
    parser.add_argument('--specify_ckpt', type=str, default='')
    parser.add_argument('--node_meta_suffix', type=str, default='')
    # logging options
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--extconf', type=str, default='')
    parser.add_argument('--expdir', type=str, default='')
    parser.add_argument('--resroot', type=str, default='../artifacts')
    args = parser.parse_args()

    mode, device, extconf, expdir, resroot = args.mode, args.device, args.extconf, args.expdir, args.resroot

    confdict = args.__dict__

    def load_extconf():
        # load from external conf file
        try:
            with open(extconf, 'r') as extconffile:
                extconfdict = yaml.load(extconffile)
                for k, v in extconfdict.items():
                    confdict[k] = v
            print('load external conf file from {}'.format(extconf))
        except:
            print('no external conf file at {}'.format(extconf))

    def load_expconf():
        # load from existing conf file
        expconfpath = os.path.join(expdir, 'conf.yaml')
        try:
            with open(expconfpath, 'r') as expconf:
                expconfdict = yaml.load(expconf)
                for k, v in expconfdict.items():
                    confdict[k] = v
            print('load existing conf file from {}'.format(expconfpath))
        except:
            print('no existing conf file at {}'.format(expconfpath))

    def make_expdir():
        return os.path.join(resroot, args.dataset, args.model + '_{}'.format(args.graph), datetime.datetime.now().isoformat())

    if extconf == '' and expdir == '':
        # a new experiment with parameters set in command line
        expdir = make_expdir()
    elif expdir == '':
        # a new experiment with parameters set in external conf file
        load_extconf()
        expdir = make_expdir()
    else:
        # continue a previous exp from checkpoint, ignore any external conf file (only use the previous conf file in expdir)
        load_expconf()
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    print(expdir)

    args = MyArgs()
    args.load_argdict(confdict)
    # mode and device are controlled by command line!
    args.mode = mode
    args.device = device
    args.extconf = extconf
    args.expdir = expdir
    args.resroot = resroot
    with open(os.path.join(expdir, 'conf.yaml'), 'w') as f:
        yaml.dump(confdict, f)

    # for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    if args.dataset.startswith('synthetic'):
        datalists = load_synthetic(dataset=args.dataset, graph=args.graph)
    elif args.dataset.startswith('noaa'):
        def parse_feature_names(feature_names):
            if len(feature_names) == 0:
                return ()
            else:
                return tuple(feature_names.split(','))

        if args.dataset.endswith('withloc'):
            with_node_meta = True
            if args.node_meta_suffix == 'utm':
                node_meta_dim = 2
            else:
                node_meta_dim = 3
        else:
            with_node_meta = False
            node_meta_dim = 0
        datalists = load_noaa(dataset=args.dataset,
                              target_features=parse_feature_names(args.noaa_target_features),
                              given_input_features=parse_feature_names(args.noaa_given_input_features),
                              graph=args.graph, with_node_meta=with_node_meta, node_meta_suffix=args.node_meta_suffix)
        mesh_matrices = datalists[4]
    elif args.dataset == 'traffic':
        datalists = load_traffic()
        node_meta_dim = 0
    # elif args.dataset == 'varicoef':
    elif args.dataset.startswith('varicoef') or args.dataset.startswith('irregular_varicoef') or args.dataset.startswith('irregular_nonlinear'):
        if args.graph == 'de':
            datalists = load_varicoef(args.dataset, args.graph)
        else:
            datalists = load_varicoef(args.dataset)
        mesh_matrices = datalists[4]
        node_meta_dim = 2
    elif args.dataset.startswith('sst'):
        if args.graph == 'de':
            datalists = load_sst(args.dataset, args.graph)
        else:
            datalists = load_sst(args.dataset)
        mesh_matrices = datalists[4]
        node_meta_dim = 2
    else:
        raise NotImplementedError('Dataset {} not implemented!'.format(args.dataset))
    if args.optim == 'LBFGS':
        train_dataloader = DataLoaderWithPad(datalists[0], batch_size=len(datalists[0]), shuffle=True, num_workers=8)
    else:
        train_dataloader = DataLoaderWithPad(datalists[0], batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    valid_dataloader = DataLoaderWithPad(datalists[1], batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
    test_dataloader = DataLoaderWithPad(datalists[2], batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
    train_normalization = datalists[3]
    print(train_normalization)

    modelnames_enable_feature_output = []
    if args.model == 'MPNN':
        model = PDGN(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            hidden_dim_pde=args.hidden_dim,
            hidden_dim_gn=args.hidden_dim,
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            mode=args.pde_mode,
            recurrent=True,
            layer_num=args.layer_num,
            gn_layer_num=2,
            edge_final_dim=args.hidden_dim,
            nophysics_mode='nopad',
            use_dist=(args.use_dist == 'True')
        )
    elif args.model == 'GraphPDE_GN_sum_notshared':
        modelnames_enable_feature_output.append(args.model)
        model = GraphPDE(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            coef_set_num_per_order=1,
            coef_net_hidden_dim=args.hidden_dim,
            coef_net_layer_num=2,
            coef_net_is_recurrent=False,
            coef_mode='abc',
            agg_mode='sum',
            coef_sharing=False,
            batchnorm='BatchNorm'
        )
    elif args.model == 'GraphPDE_GN_RGN_16_notshared':
        modelnames_enable_feature_output.append(args.model)
        model = GraphPDE(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            coef_set_num_per_order=16,
            coef_net_hidden_dim=args.hidden_dim,
            coef_net_layer_num=2,
            coef_net_is_recurrent=False,
            coef_mode='abc',
            prediction_net_hidden_dim=args.hidden_dim,
            prediction_net_layer_num=2,
            prediction_net_is_recurrent=True,
            agg_mode='RGN',
            coef_sharing=False
        )
    elif args.model == 'GraphPDE_GN_RGN_16_shared':
        modelnames_enable_feature_output.append(args.model)
        model = GraphPDE(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            coef_set_num_per_order=16,
            coef_net_hidden_dim=args.hidden_dim,
            coef_net_layer_num=2,
            coef_net_is_recurrent=False,
            coef_mode='abc',
            prediction_net_hidden_dim=args.hidden_dim,
            prediction_net_layer_num=2,
            prediction_net_is_recurrent=True,
            agg_mode='RGN',
            coef_sharing=True
        )
    elif args.model == 'SingleVAR':
        model = SingleVAR(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num
        )
    elif args.model == 'SingleMLP':
        model = SingleMLP(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            hidden_dim=args.hidden_dim,
            layer_num=args.layer_num,
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num
        )
    elif args.model == 'SingleLSTM':
        model = SingleRNN(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            hidden_dim=args.hidden_dim,
            num_layers=2,
            skip_first_frames_num=args.skip_first_frames_num
        )
    elif args.model == 'LinearRegOp_Standard':
        model = LinearRegOp(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            node_num=datalists[0][0].x.shape[0],
            optype='standard'
        )
    elif args.model == 'LinearRegOp_TriMesh':
        model = LinearRegOp(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            node_num=datalists[0][0].x.shape[0],
            optype='trimesh',
            mesh_matrices=mesh_matrices
        )
    elif args.model == 'RGNOp_Standard':
        model = LinearRegOp(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            node_num=datalists[0][0].x.shape[0],
            optype='standard',
            prediction_model='RGN',
            prediction_net_hidden_dim=args.hidden_dim,
            prediction_net_layer_num=2,
            prediction_net_is_recurrent=True
        )
    elif args.model == 'RGNOp_TriMesh':
        model = LinearRegOp(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            node_meta_dim=node_meta_dim,
            order=2,
            node_num=datalists[0][0].x.shape[0],
            optype='trimesh',
            mesh_matrices=mesh_matrices,
            prediction_model='RGN',
            prediction_net_hidden_dim=args.hidden_dim,
            prediction_net_layer_num=2,
            prediction_net_is_recurrent=True
        )
    elif args.model == 'synthetic_sum':
        modelnames_enable_feature_output.append(args.model)
        model = PDGN(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            hidden_dim_pde=args.hidden_dim,
            hidden_dim_gn=args.hidden_dim,
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            mode=args.pde_mode,
            recurrent=True,
            layer_num=args.layer_num,
            gn_layer_num=2,
            edge_final_dim=args.hidden_dim,
            nophysics_mode=None,
            use_dist=(args.use_dist == 'True'),
            pde_params_out_dim=args.pde_params_out_dim,
            use_time_grad=False,
            use_edge_grad=True,
            use_laplacian=True,
            use_pde_params=False,
            learnable_edge_grad=True,
            learnable_edge_grad_kernel_num=1,
            learnable_laplacian=True,
            learnable_laplacian_kernel_num=1,
            grad_kernel_param_loc='1',
            grad_kernel_feature='both',
            laplacian_kernel_param_loc='1',
            laplacian_kernel_feature='both',
            node_meta_dim=node_meta_dim,
            predict_model='sum'
        )
    elif args.model == 'synthetic_sum_nolearn':
        modelnames_enable_feature_output.append(args.model)
        model = PDGN(
            input_dim=datalists[0][0].x.shape[-1],
            output_dim=datalists[0][0].target.shape[-1],
            hidden_dim_pde=args.hidden_dim,
            hidden_dim_gn=args.hidden_dim,
            input_frame_num=2,
            skip_first_frames_num=args.skip_first_frames_num,
            mode=args.pde_mode,
            recurrent=True,
            layer_num=args.layer_num,
            gn_layer_num=2,
            edge_final_dim=args.hidden_dim,
            nophysics_mode=None,
            use_dist=(args.use_dist == 'True'),
            pde_params_out_dim=args.pde_params_out_dim,
            use_time_grad=False,
            use_edge_grad=True,
            use_laplacian=True,
            use_pde_params=False,
            learnable_edge_grad=True,
            learnable_edge_grad_kernel_num=1,
            learnable_laplacian=True,
            learnable_laplacian_kernel_num=1,
            grad_kernel_param_loc='1',
            grad_kernel_feature='both',
            laplacian_kernel_param_loc='1',
            laplacian_kernel_feature='dst',
            node_meta_dim=node_meta_dim,
            predict_model='sum'
        )
    else:
        raise NotImplementedError('Model {} not implemented!'.format(args.model))
    model.to(device)
    print(args.model, model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optim == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100) # not in use actually! just a placeholder

    def unpadding_output(output, data):
        list_output = unbatch_node_feature_mat(output, data.batch)
        for si in range(data.tlens.shape[0]):
            list_output[si] = list_output[si][:, :data.tlens[si], :]
        return list_output

    def unpadding_edge_output(output, data):
        edge_batch = data.batch[data.edge_index[0, :]]
        list_output = unbatch_node_feature_mat(output, edge_batch)
        for si in range(data.tlens.shape[0]):
            list_output[si] = list_output[si][:, :data.tlens[si], :]
        return list_output

    def build_list_lossfunc(lossfunc):
        def list_lossfunc(output, target, data):
            list_output = list(map(lambda x: x.flatten(0, 1), unpadding_output(output, data)))
            list_target = list(map(lambda x: x.flatten(0, 1), unpadding_output(target, data)))
            output = torch.cat(list_output, dim=0)
            target = torch.cat(list_target, dim=0)
            return lossfunc(output, target)
        return list_lossfunc

    lossfunc = build_list_lossfunc(torch.nn.MSELoss())
    errorfuncs = {
        'MAE': build_list_lossfunc(torch.nn.L1Loss()),
    }

    if args.sample_scheduler == 'always':
        sample_scheduler = AlwaysSampleScheduler()
    elif args.sample_scheduler == 'inverse_sigmoid':
        if args.invsig_ss_itern is None:
            sample_scheduler = InverseSigmoidDecaySampleScheduler(epochnum=args.epochnum, delay_start=args.invsig_delay_start)
        else:
            sample_scheduler = InverseSigmoidDecaySampleScheduler(epochnum=args.invsig_ss_itern, delay_start=args.invsig_delay_start)

    ckptdir = os.path.join(expdir, 'ckpt')
    logdir = os.path.join(expdir, 'log')
    testdir = os.path.join(expdir, 'test')
    for d in [expdir, ckptdir, logdir, testdir]:
        if not os.path.exists(d):
            os.makedirs(d)

    logf = open(os.path.join(logdir, 'log.txt'), 'a+')
    print_2way(logf, '\n{:d},{}\n'.format(os.getpid(), socket.gethostname()))
    print_2way(logf, args.__dict__)
    print_2way(logf, 'Number of params: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    writer = tensorboardX.SummaryWriter(logdir)

    start_epoch = 0
    best_valid_loss = np.inf
    best_valid_epoch = 0

    if len(args.specify_ckpt) > 0:
        specify_ckpt = args.specify_ckpt
    else:
        specify_ckpt = None
    ckpts = get_last_ckpt(ckptdir, device, suffix='_checkpoint.pt', specify=specify_ckpt)
    if ckpts['last'] is not None:
        if args.reinit_optimizer == 'True':
            last_epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler = \
                load_ckpt(model, optimizer, scheduler, ckpts['last'], restore_opt_sche=False)
        else:
            last_epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler = \
                load_ckpt(model, optimizer, scheduler, ckpts['last'], restore_opt_sche=True)
        start_epoch = last_epoch + 1

    if args.mode == 'train':
        # train
        losses_train_epoches = []
        error_train_epoches = defaultdict(list)
        losses_valid_epoches = []
        error_valid_epoches = defaultdict(list)
        for epoch in range(start_epoch, args.epochnum):
            model.train()
            losses_train_batches = []
            error_train_batches = defaultdict(list)
            for data in tqdm(train_dataloader, total=len(train_dataloader)):
                if args.optim == 'LBFGS':
                    data = data.to(device)
                    def closure(data):
                        optimizer.zero_grad()
                        output = model(data, train_sample_prob=sample_scheduler.get_train_sample_prob(epoch))
                        loss = lossfunc(output, data.target, data)
                        loss.backward()
                        return loss
                    # save last usable weights
                    torch.save(model.state_dict(), os.path.join(ckptdir, 'weight_shot.pt'))
                    while True:
                        optimizer.step(lambda: closure(data))
                        output = model(data, train_sample_prob=sample_scheduler.get_train_sample_prob(epoch))
                        loss = lossfunc(output, data.target, data)
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            break
                        else:
                            print('Reloading last okay weight and reset LBFGS optimizer...')
                            # load weight to cpu first to avoid doubling GPU memory usage
                            state_dict = torch.load(os.path.join(ckptdir, 'weight_shot.pt'), map_location='cpu')
                            model.load_state_dict(state_dict)
                            optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)
                            # remove unused results and clear GPU cache!
                            del loss
                            del output
                            torch.cuda.empty_cache()
                else:
                    optimizer.zero_grad()
                    data = data.to(device)
                    output = model(data, train_sample_prob=sample_scheduler.get_train_sample_prob(epoch))
                    loss = lossfunc(output, data.target, data)
                    loss.backward()
                    optimizer.step()

                losses_train_batches.append(loss.item())
                for en, ef in errorfuncs.items():
                    error_train_batches[en].append(ef(output, data.target, data).item())

            losses_train_epoches.append(np.mean(losses_train_batches))
            for en, ef in errorfuncs.items():
                error_train_epoches[en].append(np.mean(error_train_batches[en]))

            writer.add_scalars('loss/train', {'loss': losses_train_epoches[-1]}, epoch)
            for en, ef in errorfuncs.items():
                writer.add_scalars('error/train', {en: error_train_epoches[en][-1]}, epoch)

            if epoch % args.validint == 0:
                # validation
                model.eval()
                losses_valid_batches = []
                error_valid_batches = defaultdict(list)
                for data in valid_dataloader:
                    data = data.to(device)
                    output = model(data, train_sample_prob=0)
                    loss = lossfunc(output, data.target, data)
                    losses_valid_batches.append(loss.item())
                    for en, ef in errorfuncs.items():
                        error_valid_batches[en].append(ef(output, data.target, data).item())
                losses_valid_epoches.append(np.mean(losses_valid_batches))
                for en, ef in errorfuncs.items():
                    error_valid_epoches[en].append(np.mean(error_valid_batches[en]))
                print_2way(logf, 'Epoch {:d} Loss: train: {:.6f}, valid: {:.6f}'.format(epoch, losses_train_epoches[-1], losses_valid_epoches[-1]))
                writer.add_scalars('loss/valid', {'loss': losses_valid_epoches[-1]}, epoch)
                for en, ef in errorfuncs.items():
                    writer.add_scalars('error/valid', {en: error_valid_epoches[en][-1]}, epoch)

                if (losses_valid_epoches[-1] < best_valid_loss) or (epoch == 0):
                    if np.isnan(losses_valid_epoches[-1]):
                        losses_valid_epoches[-1] = np.inf
                    best_valid_loss = losses_valid_epoches[-1]
                    best_valid_epoch = epoch
                    save_ckpt(epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler,
                              ckptdir, prefix='best', suffix='_checkpoint.pt')

            if epoch % args.ckptint == 0:
                save_ckpt(epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler,
                              ckptdir, prefix='{:d}'.format(epoch), suffix='_checkpoint.pt')

            if epoch % args.testint == 0:
                # test
                return_features = (args.return_features == 'True') and (args.model in modelnames_enable_feature_output)
                torch.save(model.state_dict(), os.path.join(ckptdir, 'last_weight.temppt'))
                model.load_state_dict(torch.load(os.path.join(ckptdir, 'best_checkpoint.pt'), map_location=device)['model'])
                model.eval()
                losses_test_batches = []
                test_data = {
                    'output': [],
                    'target': []
                }
                if return_features:
                    test_data['features'] = {'node': [], 'edge': [], 'gradient_weight': [], 'laplacian_weight': []}
                for data in test_dataloader:
                    data = data.to(device)
                    if return_features:
                        output, output_feature = model(data, train_sample_prob=0, return_features=True)
                    else:
                        output = model(data, train_sample_prob=0)
                    loss = lossfunc(output, data.target, data)
                    losses_test_batches.append(loss.item())
                    test_data['output'].append(
                        list(map(lambda x: x.data.numpy(), unpadding_output(output.detach().cpu(), data))))
                    test_data['target'].append(
                        list(map(lambda x: x.data.numpy(), unpadding_output(data.target.detach().cpu(), data))))
                    if return_features:
                        test_data['features']['node'].append(
                            list(map(lambda x: x.data.numpy(),
                                     unpadding_output(output_feature['node'].detach().cpu(), data))))
                        for edge_feature_name in ['edge', 'gradient_weight', 'laplacian_weight']:
                            test_data['features'][edge_feature_name].append(
                                list(map(lambda x: x.data.numpy(),
                                         unpadding_edge_output(output_feature[edge_feature_name].detach().cpu(), data))))

                test_data['output'] = list(itertools.chain.from_iterable(test_data['output']))
                test_data['target'] = list(itertools.chain.from_iterable(test_data['target']))
                if return_features:
                    for feature_name in test_data['features'].keys():
                        test_data['features'][feature_name] = \
                            list(itertools.chain.from_iterable(test_data['features'][feature_name]))

                print_2way(logf,
                           'Epoch {:d} Test loss (from epoch {:d}): {:.6f}'.format(epoch, best_valid_epoch,
                                                                                   np.mean(losses_test_batches)))
                writer.add_scalars('loss/test', {'loss': np.mean(losses_test_batches)}, best_valid_epoch)
                np.save(os.path.join(testdir, '{:d}_test.npy'.format(epoch)), test_data)
                np.save(os.path.join(testdir, 'normalization.npy'), train_normalization)
                model.load_state_dict(torch.load(os.path.join(ckptdir, 'last_weight.temppt'), map_location=device))

    # test
    return_features = (args.return_features == 'True') and (args.model in modelnames_enable_feature_output)
    model.load_state_dict(torch.load(os.path.join(ckptdir, 'best_checkpoint.pt'), map_location=device)['model'])
    model.eval()
    losses_test_batches = []
    test_data = {
        'output': [],
        'target': []
    }
    if return_features:
        test_data['features'] = {'node': [], 'edge': [], 'gradient_weight': [], 'laplacian_weight': []}
    for data in test_dataloader:
        data = data.to(device)
        if return_features:
            output, output_feature = model(data, train_sample_prob=0, return_features=True)
        else:
            output = model(data, train_sample_prob=0)
        loss = lossfunc(output, data.target, data)
        losses_test_batches.append(loss.item())
        test_data['output'].append(
            list(map(lambda x: x.data.numpy(), unpadding_output(output.detach().cpu(), data))))
        test_data['target'].append(
            list(map(lambda x: x.data.numpy(), unpadding_output(data.target.detach().cpu(), data))))
        if return_features:
            test_data['features']['node'].append(
                list(map(lambda x: x.data.numpy(),
                         unpadding_output(output_feature['node'].detach().cpu(), data))))
            for edge_feature_name in ['edge', 'gradient_weight', 'laplacian_weight']:
                test_data['features'][edge_feature_name].append(
                    list(map(lambda x: x.data.numpy(),
                             unpadding_edge_output(output_feature[edge_feature_name].detach().cpu(), data))))

    test_data['output'] = list(itertools.chain.from_iterable(test_data['output']))
    test_data['target'] = list(itertools.chain.from_iterable(test_data['target']))
    if return_features:
        for feature_name in test_data['features'].keys():
            test_data['features'][feature_name] = \
                list(itertools.chain.from_iterable(test_data['features'][feature_name]))

    print_2way(logf, 'Test loss (from epoch {:d}): {:.6f}'.format(best_valid_epoch, np.mean(losses_test_batches)))
    writer.add_scalars('loss/test', {'loss': np.mean(losses_test_batches)}, best_valid_epoch)
    np.save(os.path.join(testdir, 'test.npy'), test_data)
    np.save(os.path.join(testdir, 'normalization.npy'), train_normalization)

    logf.close()


if __name__ == '__main__':
    main()

from trainer import Trainer
import numpy as np
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
from torch.distributed.elastic.multiprocessing.errors import record


@record
@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    print("args: DictConfig", args)
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    assert args.epochs is not None
    assert args.batch_size is not None
    assert args.checkname is not None


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = Trainer(args)

    if args.mode == 'train':
        train(trainer, args)
    elif args.mode == 'eval':
        trainer.test1(epoch=0)


    if trainer.cuda_device == 0:
        trainer.writer.close()

    if args.distributed and dist.get_rank() == 0:
        dist.destroy_process_group()



def train(trainer, args):
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.test(epoch=epoch)

if __name__ == "__main__":
    main()

from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import dycl.lib as lib

from .checkpoint import checkpoint
from .accuracy_calculator import evaluate
from .base_training_loop_our import base_training_loop_our


def TrainDyCL(
    config,#config config/default.yaml
    log_dir,
    net,
    criterion,
    optimizer,
    scheduler,
    scaler,
    acc,
    train_dts,
    #train_dts_2,
    #train_dts_3,
    val_dts,
    test_dts_1,
    #test_dts_2,
    test_dts_3,
    sampler,
    #sampler_2,
    #sampler_3,
    writer,
    restore_epoch,#0
):
    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    lib.LOGGER.info(f"Training of model {config.experience.experiment_name}")# ???

    metrics = None
    for e in range(1 + restore_epoch, config.experience.max_iter + 1):# [1,101)

        lib.LOGGER.info(f"Training : @epoch #{e} for model {config.experience.experiment_name}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        #train_dts={}
        train_dts=train_dts

        test_dts={}
        test_dts['satellite']=test_dts_1
        #test_dts['street']=test_dts_2
        test_dts['drone']=test_dts_3

        loaders =  DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=config.experience.num_workers,#  10
            pin_memory=config.experience.pin_memory # true
        )
            
        
        """
        train_dataloader = DataLoader(train_dts,
                                  batch_size=80,
                                  num_workers=4,
                                  shuffle=False,
                                  pin_memory=True)
        """
        logs = base_training_loop_our(
            config=config,
            net=net,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=e,
        )

        if (
            (config.experience.warmup_step is not None)
            and (config.experience.warmup_step >= e)
        ):
            pass
        else:
            for sch in scheduler["on_epoch"]:
                sch.step()

            for crit, _ in criterion:
                if hasattr(crit, 'on_epoch'):
                    crit.on_epoch()

        end_train_time = time()

        dataset_dict = {}

        if (config.experience.train_eval_freq > -1) and ((e % config.experience.train_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["train"] = train_dts

        if (config.experience.val_eval_freq > -1) and ((e % config.experience.val_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["val"] = val_dts

        if (config.experience.test_eval_freq > -1) and ((e % config.experience.test_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["satellite_drone"]={}
            dataset_dict["drone_satellite"]={}
            # dataset_dict["drone_drone"]={}
            # dataset_dict["satellite_satellite"]={}
            dataset_dict["satellite_drone"]["query"] = test_dts['satellite']
            dataset_dict["satellite_drone"]["gallery"] = test_dts['drone']
            dataset_dict["drone_satellite"]["query"] = test_dts['drone']
            dataset_dict["drone_satellite"]["gallery"] = test_dts['satellite']


        metrics = None
        if dataset_dict:
            metrics = evaluate(
                net=net,
                dataset_dict=dataset_dict,
                acc=acc,
                epoch=e,
            )
            torch.cuda.empty_cache()

        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, opt in optimizer.items():
            writer.add_scalar(f"LR/{grp}", list(lib.get_lr(opt).values())[0], e)

        for k, v in logs.items():
            lib.LOGGER.info(f"{k} : {v:.4f}")
            writer.add_scalar(f"DyCL/Train/{k}", v, e)

        if metrics is not None:
            for split, mtrc in metrics.items():
                for k, v in mtrc.items():
                    if k == 'epoch':
                        continue
                    lib.LOGGER.info(f"{split} --> {k} : {np.around(v*100, decimals=2)}")
                    writer.add_scalar(f"DyCL/{split.title()}/Evaluation/{k}", v, e)
                print()

        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)
        elapsed_time_eval = lib.format_time(end_time - end_train_time)

        lib.LOGGER.info(f"Epoch took : {elapsed_time}")
        if metrics is not None:
            lib.LOGGER.info(f"Training loop took : {elapsed_time_train}")
            lib.LOGGER.info(f"Evaluation step took : {elapsed_time_eval}")

        print()
        print()

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        checkpoint(
            log_dir=log_dir,
            save_checkpoint=(e % config.experience.save_model == 0) or (e == config.experience.max_iter),
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            epoch=e,
            config=config,
            metrics=metrics,
        )

    return metrics
import os

import torch
from tqdm import tqdm

import dycl.lib as lib


def _calculate_loss_and_backward(
    config,
    net,
    batch,
    #batch3,
    relevance_fn_1,
    #relevance_fn_3,
    matrix,
    criterion,
    optimizer,
    scaler,
    epoch,
):
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        di, di3 = net(batch["image_s"].cuda(), batch["image_d"].cuda())

        labels = batch["label"].cuda()
        coords = batch["coord"].cuda()
        index = batch["index"].cuda()

        

        logs = {}
        losses = []

        for crit, weight in criterion: 
            if crit.__class__.__name__ == "DyCLLoss":
             loss = crit(
                di,
                di3,
                labels,
                matrix,
             ) 
            
            if crit.__class__.__name__ == "CSLLoss":
              loss1 = crit(
                di,
                labels,
                matrix,
             ) 
              loss3 = crit(
                di,
                labels,
                matrix,

             ) 
              loss1 = loss1.mean()
              loss3 = loss3.mean()
              loss = (loss1 + loss3)/2             

            if crit.__class__.__name__ == "HAPLoss":
              loss1 = crit(
                di,
                di3,
                labels,
                matrix,
                relevance_fn=relevance_fn_1,
                indexes = index
             ) 
              loss3 = crit(
                di3,
                di,
                labels,
                matrix,
                relevance_fn=relevance_fn_1,
                indexes = index
             ) 
              loss1 = loss1.mean()
              loss3 = loss3.mean()
              loss = (loss1 + loss3)/2

            if crit.__class__.__name__ == "ClusterLoss":
              loss1 = crit(
                di,
                labels,
                relevance_fn=relevance_fn_1,
                indexes = index
             ) 
             
              loss3 = crit(
                di3,
                labels,
                relevance_fn=relevance_fn_1,
                indexes = index
             ) 
              loss1 = loss1.mean()
              loss3 = loss3.mean()
              loss = (loss1 + loss3)/2
            
            
            losses.append(weight * loss)

            logs[f"{crit.__class__.__name__}_{crit.hierarchy_level}"] = loss.item()         
          
    total_loss = sum(losses)
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_training_loop_our(
    config,
    net,
    loaders,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
):
    meter = lib.DictAverage()
    
    net.train()
    net.zero_grad()

    
    iterator1 = tqdm(loaders, disable=os.getenv('TQDM_DISABLE'))

    #i=-1
    for i, batch in enumerate(iterator1):                                  #for batch3 in zip(iterator3):
        #i=i+1
        logs = _calculate_loss_and_backward(
            config,
            net,
            batch,
            loaders.dataset.compute_relevance_on_the_fly,
            loaders.dataset.matrix,
            criterion,
            optimizer,
            scaler,
            epoch,
        )

        if config.experience.record_gradient:
            if scaler is not None:
                for opt in optimizer.values():
                    scaler.unscale_(opt)

            logs["gradient_norm"] = lib.get_gradient_norm(net)

        if config.experience.gradient_clipping_norm is not None:
            if (scaler is not None) and (not config.experience.record_gradient):
                for opt in optimizer.values():
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.experience.gradient_clipping_norm)

        for key, opt in optimizer.items():
            if (
                (config.experience.warmup_step is not None)
                and (config.experience.warmup_step >= epoch)
                and (key in config.experience.warmup_keys)
            ):
                if i == 0:
                    lib.LOGGER.warning("Warmimg UP")
                continue
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)

        for crit, _ in criterion:
            if hasattr(crit, 'update'):
                crit.update(scaler)

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        if scaler is not None:
            scaler.update()

        meter.update(logs)
        
        if not os.getenv('TQDM_DISABLE'):
            iterator1.set_postfix(meter.avg)
        else:
            if (i + 1) % config.experience.print_freq == 0:
                lib.LOGGER.info(f'Iteration : {i}/{len(loaders)}')
                for k, v in logs.items():
                    lib.LOGGER.info(f'Loss: {k}: {v} ')
        
       

    for crit, _ in criterion:
        if hasattr(crit, 'optimize_proxies'):
            crit.optimize_proxies(loaders['satellite'].dataset.compute_relevance_on_the_fly)

    return meter.avg
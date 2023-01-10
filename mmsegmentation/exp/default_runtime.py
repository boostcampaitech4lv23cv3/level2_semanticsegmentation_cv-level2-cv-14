# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='MMSegWandbHook',
            init_kwargs={
                "project": "segmentation",
                "name": "base model",
                "entity": "level2_object-detection-cv14",
            },
            interval=100,
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            num_eval_images=0,
        )
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

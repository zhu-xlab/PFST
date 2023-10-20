# Obtained from: https://github.com/open-mmlab/dasegmentation/tree/v0.16.0

# optimizer
optimizer = dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

# LTU 2 things to check

- torchrun
- model.print_trainable_parameters()
- peft whole model
- padding left to allow batching. huh?
- pad token to unk: why?
- ds: explicit train_test_split and shuffle: do we need this?
- why? `model.is_parallelizable = model.model_parallel = True`

# dsl
NN for DSL-1/2 devices 
Training and evaluation scripts

# Contains
- arch_nn_hyper.py
` looking for the best NN architecture within grid search and dropout levels`

- make_experiment.sh
` Cross validation experiment to estimate bottom edge of sensitivity and specificity`

- one_step_split_train_test_validation
` tunable train/test/validation split`

- predict.py
` predict on the *best-by-valacc* model, sensitivity and specificity`

- train.py
` low-epochs train with checkpoints and early-stopping`


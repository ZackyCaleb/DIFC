import argparse
import torch
from solver import Solver
import os
from recordermeter import RecorderMeter
def main(model_cofig = None):
    model_cofig = {
        "batch_size": 64,
        "lr": 1e-4,
        "momentum": 0.9,
        "max_epoch": 40,
        "num_k": 4,
        "size": 112,

        "sour_path": r'F:\FER_dataset_clearned\RAF_DB\test',
        # "target_train_path": r'F:\FER_dataset_clearned\jaffe\train',
        "target_train_path": r'F:\FER_dataset_clearned\SFEW\RetinaFace_sfew2.0\train',
        # "target_test_path": r'F:\FER_dataset_clearned\jaffe\test',
        "target_test_path": r'F:\FER_dataset_clearned\SFEW\RetinaFace_sfew2.0\valid',
        "mode": 'sfew'}

    eng = Solver(num_k=model_cofig['num_k'], batch_size=model_cofig['batch_size'], lr=model_cofig['lr'], momentum=model_cofig['momentum'],
                 size=model_cofig['size'], sour_path=model_cofig['sour_path'], target_train_path=model_cofig['target_train_path'],
                 target_test_path=model_cofig['target_test_path'], mode=model_cofig['mode'])

    for t in range(model_cofig['max_epoch']):
        eng.train(t)
        eng.test(t)


if __name__ == '__main__':
    main()

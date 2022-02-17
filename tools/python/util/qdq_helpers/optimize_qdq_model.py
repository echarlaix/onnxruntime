#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import onnx
import os
import pathlib

from .qdq_model_utils import fix_dq_nodes_with_multiple_consumers


def optimize_qdq_model():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='''
                                     Run utilities to make a QDQ format model more consistent and compatible with the
                                     ONNX spec in order to ensure optimal performance when executed using
                                     ONNX Runtime.
                                     ''')

    parser.add_argument('input_model', type=pathlib.Path, help='Provide path to ONNX model to update.')
    parser.add_argument('output_model', type=pathlib.Path, help='Provide path to write updated ONNX model to.')

    args = parser.parse_args()

    model = onnx.load(str(args.input_model.resolve(strict=True)))

    # currently there's just one utility to run
    fix_dq_nodes_with_multiple_consumers(model)

    onnx.save(model, str(args.output_model.resolve()))


if __name__ == '__main__':
    optimize_qdq_model()
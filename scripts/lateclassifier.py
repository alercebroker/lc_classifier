#!/usr/bin/env python

import click
import os


@click.group()
def cli():
    """
    ALeRCE Late Classifier.
    """
    pass


@cli.command()
@click.argument('detections_dir', type=click.Path(exists=True))
@click.argument('non_detections_dir', type=click.Path(exists=True))
@click.argument('output_dir', default="features", type=click.Path())
def compute_features(detections_dir, non_detections_dir, output_dir):
    print("dsajhdkas")


if __name__ == '__main__':
    cli()
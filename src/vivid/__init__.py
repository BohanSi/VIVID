"""VIVID: stepwise generation of spatial omics slices from aligned endpoints."""
from .api import (Session, read_inputs, configure, select_image_size, encode,
                  check_backend, interpolate, reconstruct, save_results)
from .series import SeriesSession
from .examples import read_mb1, preprocess_inputs
from .alignment import align_inputs
from .plotting import (plot_inputs, plot_encoding, plot_interpolation,
                       plot_predictions, plot_feature)

__version__ = "0.1.0"
__all__ = ["SeriesSession", "preprocess_inputs", "read_mb1", "Session", "read_inputs", "configure", "select_image_size", "encode",
           "check_backend", "interpolate", "reconstruct", "save_results", "align_inputs",
           "plot_inputs", "plot_encoding", "plot_interpolation", "plot_predictions", "plot_feature"]

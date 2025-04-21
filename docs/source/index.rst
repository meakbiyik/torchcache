.. torchcache documentation master file

Welcome to torchcache!
======================

`torchcache` offers an effortless way to cache PyTorch module outputs on-the-fly. By caching the outputs of a module, you can save time and resources when running the same pre-trained model on the same inputs multiple times.

Note that gradients will not flow through the cached outputs.

Citation
--------

If you use our work, please consider citing our paper:

.. code-block:: bibtex

   @inproceedings{akbiyik2023routeformer,
       title={Leveraging Driver Field-of-View for Multimodal Ego-Trajectory Prediction},
       author={M. Eren Akbiyik, Nedko Savov, Danda Pani Paudel, Nikola Popovic, Christian Vater, Otmar Hilliges, Luc Van Gool, Xi Wang},
       booktitle={International Conference on Learning Representations},
       year={2025}
   }

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   installation
   usage
   how_it_works
   environment_variables
   contribution
   api

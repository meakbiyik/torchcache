"""Tests for the caching module."""
import os

import pytest
import torch
import torch.nn as nn

from torchcache import TorchCache
from torchcache.torchcache import _TorchCache


# Test Module
class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()

    def forward(self, x):
        return x * 2


# Test basic functionality of the caching mechanism without persistent storage.
def test_basic_caching():
    @TorchCache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # Assert that the cache is empty
    assert len(model.cache_instance.cache) == 0

    # First pass, caching should occur
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Assert that the cache is not empty
    assert len(model.cache_instance.cache) == input_tensor.shape[0]

    # Check that the hash of the input tensor is in the cache
    hashes = model.cache_instance.hash_tensor(input_tensor)
    for hash in hashes:
        assert hash.item() in model.cache_instance.cache

    # Second pass, should retrieve from cache but result should be the same
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)


# Test caching mechanism with persistent storage.
def test_persistent_caching(tmp_path):
    @TorchCache(persistent=True, persistent_cache_dir=tmp_path)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # First pass, caching should occur and save to file
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Check if cache files were created
    assert len(list(tmp_path.iterdir())) == 2

    # Second pass, should retrieve from cache from memory
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)

    # Now create a new instance of the model and check if the cache is loaded from disk
    # We re-define the class to flush the cache
    @TorchCache(persistent=True, persistent_cache_dir=tmp_path)
    class CachedModule(SimpleModule):
        pass

    model2 = CachedModule()
    original_load_from_file = model2.cache_instance._load_from_file
    model2.cache_instance.original_load_from_file = original_load_from_file
    load_from_file_called = False

    def _load_from_file(*args, **kwargs):
        nonlocal load_from_file_called
        load_from_file_called = True
        original_load_from_file(*args, **kwargs)

    model2.cache_instance._load_from_file = _load_from_file
    output_cached = model2(input_tensor)
    assert torch.equal(output, output_cached)
    assert load_from_file_called


# Test temporary cachedir
def test_persistent_caching_temporary_cachedir():
    @TorchCache(persistent=True)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    temporary_path = model.cache_instance.cache_dir

    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # First pass, caching should occur and save to file
    output = model(input_tensor)
    assert torch.equal(output, input_tensor * 2)

    # Check if cache files were created
    assert len(list(temporary_path.iterdir())) == 2

    # Second pass, should retrieve from cache but result should be the same
    output_cached = model(input_tensor)
    assert torch.equal(output, output_cached)


# Test hashing functionality.
def test_hashing():
    cache = _TorchCache()
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    hashes = cache.hash_tensor(input_tensor)

    assert hashes.shape[0] == input_tensor.shape[0]


# Test for mixed cache hits
def test_mixed_cache_hits():
    @TorchCache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    input_tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    # First pass for caching
    model(input_tensor1)
    # Second pass where one tensor is cached and the other isn't
    combined_output = model(torch.cat([input_tensor1, input_tensor2], dim=0))

    assert torch.equal(combined_output[:2], input_tensor1 * 2)
    assert torch.equal(combined_output[2:], input_tensor2 * 2)


# Test monkey patching
def test_monkey_patching():
    @TorchCache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    # First pass for caching
    model(input_tensor)
    # Monkey-patching should be applied here
    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


# Module with multiple input tensors
class DoubleInputModule(nn.Module):
    def forward(self, x, y):
        return x * 2 + y * 3


def test_multiple_inputs():
    @TorchCache(persistent=False)
    class CachedModule(DoubleInputModule):
        pass

    model = CachedModule()
    input_tensor1 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    input_tensor2 = torch.tensor([[4, 5, 6]], dtype=torch.float32)

    output = model(input_tensor1, input_tensor2)

    assert torch.equal(output, input_tensor1 * 2 + input_tensor2 * 3)

    # Second pass, should retrieve from cache but result should be the same
    output = model(input_tensor1, input_tensor2)

    assert torch.equal(output, input_tensor1 * 2 + input_tensor2 * 3)


# Test different tensor shapes
def test_different_shapes():
    @TorchCache(persistent=False)
    class CachedModule(SimpleModule):
        pass

    model = CachedModule()
    input_tensor = torch.randn((5, 5, 5, 5))

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)


# Test overriding TorchCache arguments with module attributes that starts with "torchcache_"
def test_override_torchcache_args():
    @TorchCache(persistent=False)
    class CachedModule(SimpleModule):
        def __init__(self):
            super().__init__()
            self.torchcache_persistent = True

    model = CachedModule()
    input_tensor = torch.randn((5, 5, 5, 5))

    output = model(input_tensor)

    assert torch.equal(output, input_tensor * 2)
    assert model.cache_instance.persistent


# Test training with forward and backward to ensure that
# there are no missing detach calls.
def test_training():
    @TorchCache(persistent=False)
    class CachedModule(SimpleModule):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 5))

        def forward(self, x):
            return x @ self.weight

    model = CachedModule()
    input_tensor = torch.randn((1, 5))

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    assert torch.equal(output, input_tensor @ model.weight)
    assert model.weight.grad is not None

    # run again to ensure that the graph is sane
    input_tensor = torch.randn((1, 5))
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Cleanup after all tests run, ensuring no cache files are left behind
    for f in os.listdir():
        if f.endswith(".pt.br"):
            os.remove(f)

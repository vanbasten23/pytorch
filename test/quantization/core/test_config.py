# Owner(s): ["oncall: quantization"]

import torch
import torch.nn.intrinsic as nni
from torch.testing._internal.common_quantization import QuantizationTestCase

from torch.ao.quantization.backend_config.backend_config import *
from torch.ao.quantization.backend_config.observation_type import ObservationType


class TestBackendConfig(QuantizationTestCase):

    # =============
    #  DtypeConfig
    # =============

    dtype_config1 = DtypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float
    )

    dtype_config2 = DtypeConfig(
        input_dtype=torch.float16,
        output_dtype=torch.float,
        is_dynamic=True
    )

    dtype_config_dict1 = { 
        "input_dtype": torch.quint8,
        "output_dtype": torch.quint8,
        "weight_dtype": torch.qint8,
        "bias_dtype": torch.float,
    } 

    dtype_config_dict2 = { 
        "input_dtype": torch.float16,
        "output_dtype": torch.float,
        "is_dynamic": True,
    }

    def test_dtype_config_from_dict(self):
        assert(DtypeConfig.from_dict(self.dtype_config_dict1) == self.dtype_config1)
        assert(DtypeConfig.from_dict(self.dtype_config_dict2) == self.dtype_config2)

    def test_dtype_config_to_dict(self):
        assert(self.dtype_config1.to_dict() == self.dtype_config_dict1)
        assert(self.dtype_config2.to_dict() == self.dtype_config_dict2)

    # =================
    #  BackendOpConfig
    # =================

    def test_backend_op_config_set_pattern(self):
        conf = BackendOpConfig(torch.nn.Linear)
        assert(conf.pattern == torch.nn.Linear)
        conf.set_pattern((torch.nn.ReLU, torch.nn.Conv2d))
        assert(conf.pattern == (torch.nn.ReLU, torch.nn.Conv2d))

    def test_backend_op_config_set_observation_type(self):
        conf = BackendOpConfig(torch.nn.Linear)
        assert(conf.observation_type == ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        conf.set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        assert(conf.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

    def test_backend_op_config_set_dtype_configs(self):
        conf = BackendOpConfig(torch.nn.Linear)
        assert(len(conf.dtype_configs) == 0)
        conf.set_dtype_configs([self.dtype_config1, self.dtype_config2])
        assert(len(conf.dtype_configs) == 2)
        assert(conf.dtype_configs[0] == self.dtype_config1)
        assert(conf.dtype_configs[1] == self.dtype_config2)

    def test_backend_op_config_set_root_module(self):
        conf = BackendOpConfig(nni.LinearReLU)
        assert(conf.root_module is None)
        conf.set_root_module(torch.nn.Linear)
        assert(conf.root_module == torch.nn.Linear)

    def test_backend_op_config_set_qat_module(self):
        pass

    def test_backend_op_config_set_quantized_reference_module(self):
        pass

    def test_backend_op_config_set_fuser_module(self):
        pass

    def test_backend_op_config_set_fuser_method(self):
        pass

    def test_backend_op_config_set_root_node_getter(self):
        pass

    def test_backend_op_config_set_extra_inputs_getter(self):
        pass

    def test_backend_op_config_set_num_tensor_args_to_observation_type(self):
        pass

    def test_backend_op_config_set_input_type_to_index(self):
        pass

    def test_backend_op_config_set_input_output_observed(self):
        pass

    def test_backend_op_config_set_overwrite_output_fake_quantize(self):
        pass

    def test_backend_op_config_set_overwrite_output_observer(self):
        pass

    def test_backend_op_config_from_dict(self):
        pass

    def test_backend_op_config_to_dict(self):
        pass

    # ===============
    #  BackendConfig
    # ===============

    def test_backend_config_set_name(self):
        pass

    def test_backend_config_set_config(self):
        pass

    def test_backend_config_from_dict(self):
        pass

    def test_backend_config_to_dict(self):
        pass


if __name__ == '__main__':
    raise RuntimeError("This _test file is not meant to be run directly, use:\n\n"
                       "\tpython _test/_test_quantization.py TESTNAME\n\n"
                       "instead.")

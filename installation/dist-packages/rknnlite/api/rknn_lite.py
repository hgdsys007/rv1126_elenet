# -*- coding:utf-8 -*-

import os
import copy
import traceback
import platform
from .rknn_platform_utils import get_host_os_platform, get_librknn_api_require_dll_dir, get_ntb_devices, \
    get_adb_devices, list_support_target_platform
from .rknn_runtime import RKNNRuntime
from .rknn_perf import collect_memory_detail, format_memory_detail
from .rknn_log import set_log_level_and_file_path
from .npu_config.cpu_npu_mapper import get_support_target_soc


class RKNNLite:
    """
    Rockchip NN Kit
    """
    def __init__(self, verbose=False, verbose_file=None):
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if get_host_os_platform() == 'Windows_x64':
            require_dll_dir = get_librknn_api_require_dll_dir()
            new_path = os.environ["PATH"] + ";" + require_dll_dir
            os.environ["PATH"] = new_path
        self.target = 'simulator'
        self.verbose = verbose
        if verbose_file is not None:
            if os.path.dirname(verbose_file) != "" and not os.path.exists(os.path.dirname(verbose_file)):
                verbose_file = None
        self.rknn_log = set_log_level_and_file_path(verbose, verbose_file)

        if verbose:
            if verbose_file is None:
                self.rknn_log.w('Verbose file path is invalid, debug info will not dump to file.')
            else:
                self.rknn_log.d('Save log info to: {}'.format(verbose_file))
        self.rknn_data = None
        self.load_model_in_npu = False
        self.rknn_runtime = None
        self.root_dir = cur_path

    def load_rknn(self, path, load_model_in_npu=False):
        """
        Load RKNN model
        :param path: RKNN model file path
        :param load_model_in_npu: load model in npu, if set True, must run on device with NPU
        :return: success: 0, failure: -1
        """
        if not load_model_in_npu and not os.path.exists(path):
            self.rknn_log.e('Invalid RKNN model path: {}'.format("None" if (path is None or path == "") else path),
                            False)
            return -1
        try:
            # load model in npu
            if load_model_in_npu:
                self.load_model_in_npu = load_model_in_npu
                self.rknn_data = path.encode(encoding='utf-8')
                return 0

            # Read RKNN model file data
            with open(path, 'rb') as f:
                self.rknn_data = f.read()
        except:
            self.rknn_log.e('Catch exception when loading RKNN model [{}]!'.format(path), False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        if self.rknn_data is None:
            return -1

        return 0

    def list_devices(self):
        """
        print all adb devices and devices use ntb.
        :return: adb_devices, list; ntb_devices, list. example:
                 adb_devices = ['0123456789ABCDEF']
                 ntb_devices = ['TB-RK1808S000000009']
        """
        # get adb devices
        adb_devices = get_adb_devices()
        # get ntb devices
        ntb_devices = get_ntb_devices()
        adb_devices_copy = copy.deepcopy(adb_devices)
        for device in adb_devices_copy:
            if device in ntb_devices:
                adb_devices.remove(device)
        self.rknn_log.p('*' * 25)
        if len(adb_devices) > 0:
            self.rknn_log.p('all device(s) with adb mode:')
            self.rknn_log.p(",".join(adb_devices))
        if len(ntb_devices) > 0:
            self.rknn_log.p('all device(s) with ntb mode:')
            self.rknn_log.p(",".join(ntb_devices))
        if len(adb_devices) == 0 and len(ntb_devices) == 0:
            self.rknn_log.p('None devices connected.')
        self.rknn_log.p('*' * 25)
        if len(adb_devices) > 0 and len(ntb_devices) > 0:
            all_adb_devices_are_ntb_also = True
            for device in adb_devices:
                if device not in ntb_devices:
                    all_adb_devices_are_ntb_also = False
            if not all_adb_devices_are_ntb_also:
                self.rknn_log.w('Cannot use both device with adb mode and device with ntb mode.')
        return adb_devices, ntb_devices

    def init_runtime(self, target=None, target_sub_class=None, device_id=None, perf_debug=False, eval_mem=False,
                     async_mode=False, rknn2precompile=False):
        """
        Init run time environment. Needed by called before inference or eval performance.
        :param target: target platform, RK1808/RK1109/RK1126 or RK3399Pro. None means simulator on Liunx_x86, means NPU on Linux_aarch64
        :param target_sub_class: sub class of target, now we only have AI Compute Stick, sub class of RK1808,
                                 value is 'AICS'.
        :param device_id: adb device id, only needed when multiple devices connected to pc
        :param perf_debug: enable or disable debugging performance, it will affect performance
        :param eval_mem: enable or disable debugging memory usage, it will affect performance
        :param async_mode: enable or disable async mode
        :param rknn2precompile: convert rknn model to precompile mode
        :return: success: 0, failure: -1
        """
        if target is None and platform.machine() != 'aarch64' and platform.machine() != 'armv7l':
            self.rknn_log.e("RKNN Toolkit Lite does not support simulator, please specify the target: RK1808/RK1109/RK1126 or RK3399Pro.", False)
            return -1

        if self.rknn_data is None:
            self.rknn_log.e("Model is not loaded yet, this interface should be called after load_rknn!", False)
            return -1

        if rknn2precompile:
            self.rknn_log.w('The rknn2precompile is not currently supported on RKNN Toolkit Lite.')
        if perf_debug:
            self.rknn_log.w("The perf_debug is not currently supported on RKNN Toolkit Lite.")

        # if rknn_runtime is not None, release it first
        if self.rknn_runtime is not None:
            self.rknn_runtime.release()
            self.rknn_runtime = None

        try:
            self.rknn_runtime = RKNNRuntime(root_dir=self.root_dir, target=target, target_sub_class=target_sub_class,
                                            device_id=device_id, perf_debug=perf_debug, eval_memory=eval_mem,
                                            async_mode=async_mode, rknn2precompile=rknn2precompile)
        except:
            self.rknn_log.e('Catch exception when init runtime!', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        if perf_debug and (self.rknn_runtime.host is not None or self.rknn_runtime.target_soc is not None):
            self.rknn_log.w('Flag perf_debug has been set, it will affect the performance of inference!')
        if eval_mem and (self.rknn_runtime.host is not None or self.rknn_runtime.target_soc is not None):
            self.rknn_log.w('Flag eval_mem has been set, it will affect the performance of inference!')

        try:
            self.rknn_runtime.build_graph(self.rknn_data, self.load_model_in_npu)
        except:
            self.rknn_log.e('Catch exception when init runtime!', False)
            if target is not None and target.upper() in get_support_target_soc() and platform.machine() != "armv7l":
                adb_devices, ntb_devices = self.list_devices()
                for device in adb_devices:
                    if device in ntb_devices:
                        adb_devices.remove(device)
                devices = adb_devices + ntb_devices
                self.rknn_log.e('{}'.format(devices), False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        return 0

    def inference(self, inputs, data_type=None, data_format=None, inputs_pass_through=None, get_frame_id=False):
        """
        Run model inference
        :param inputs: Input data List (ndarray list)
        :param data_type: Data type (str), currently support: int8, uint8, int16, float16, float32, default uint8
        :param data_format: Data format (str), current support: 'nhwc', 'nchw', default is 'nhwc'
        :param inputs_pass_through: set pass_through flag(0 or 1: 0 meas False, 1 means True) for every input. (list)
        :param get_frame_id: weather need get output/input frame id when using async mode,it can be use in camera demo
        :return: Output data (ndarray list)
        """
        if self.rknn_runtime is None:
            self.rknn_log.e('Runtime environment is not inited, please call init_runtime to init it first!', False)
            return None

        # set inputs
        try:
            self.rknn_runtime.set_inputs(inputs, data_type, data_format, inputs_pass_through=inputs_pass_through)
        except:
            self.rknn_log.e('Catch exception when setting inputs.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        # run
        try:
            ret = self.rknn_runtime.run(get_frame_id)
        except:
            self.rknn_log.e('Catch exception when running RKNN model.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        # get outputs
        try:
            outputs = self.rknn_runtime.get_outputs(get_frame_id)
        except:
            self.rknn_log.e('Catch exception when getting outputs.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        if not get_frame_id:
            return outputs
        else:
            outputs.append(ret[1])
            return outputs

    def eval_memory(self, is_print=True):
        """
        Get memory usage to evaluate memory loss.
        :param is_print: Format print memory usage
        :return: memory_detail (Dict)
        """
        if self.rknn_runtime is None:
            self.rknn_log.e('Runtime environment is not inited, please call init_runtime to init it first!', False)
            return None

        try:
            detail_len = self.rknn_runtime.get_memory_detail_len()
        except:
            self.rknn_log.e('Catch exception when getting memory detail length.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None
        if detail_len == 0:
            self.rknn_log.e('Get memory detail success, but detail is empty!', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None
        try:
            memory_detail_str = self.rknn_runtime.get_memory_detail(detail_len)
        except:
            self.rknn_log.e('Catch exception when getting memory detail.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        if len(memory_detail_str) != detail_len:
            self.rknn_log.e('E Catch exception when getting memory detail, len of memory_detail_str {} vs {}'.format(
                len(memory_detail_str), detail_len), False)
            return None

        try:
            memory_detail = collect_memory_detail(memory_detail_str)
        except:
            self.rknn_log.e('Catch exception when collecting performance detail.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        if is_print and memory_detail is not None:
            self.rknn_log.p(format_memory_detail(self.rknn_data, memory_detail))

        return memory_detail

    def get_sdk_version(self):
        """
        Get SDK version
        :return: sdk_version
        """
        if self.rknn_runtime is None:
            self.rknn_log.e('Runtime environment is not inited, please call init_runtime to init it first!', False)
            return None

        try:
            sdk_version, _, _ = self.rknn_runtime.get_sdk_version()
        except Exception:
            self.rknn_log.e('Catch exception when get sdk version', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        return sdk_version

    def list_support_target_platform(self, rknn_model=None):
        """
        List all target platforms which can run the model in rknn_model.
        :param rknn_model: RKNN model path, if None, all target platforms will be printed, and ordered by NPU model.
        :return: support_target(dict)
        """
        if rknn_model is not None and not os.path.exists(rknn_model):
            self.rknn_log.e('The model {} does not exist.'.format(rknn_model))
            return None
        return list_support_target_platform(rknn_model)

    def release(self):
        """
        Release RKNN resource
        :return: None
        """
        # release rknn runtime
        if self.rknn_runtime is not None:
            self.rknn_runtime.release()
            self.rknn_runtime = None


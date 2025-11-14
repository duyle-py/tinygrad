#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/csrc/PyInterpreter.h>
#include <ATen/OpaqueTensorImpl.h>
#include <iostream>
#include <cstdlib>

// Logging helper (outside namespace)
static bool get_debug_logs() {
  static bool debug = std::getenv("TORCH_DEBUG") != nullptr;
  return debug;
}

#define LOG_DEBUG(msg) if (get_debug_logs()) std::cerr << "[wrapped_tensor] " << msg << std::endl

// register guard
namespace at {
namespace detail {
// NOTE: pytorch's no-op class throws error on backwards with events/streams
// TODO: why are there events in autograd?
struct CustomNoOpDeviceGuardImpl : public c10::impl::DeviceGuardImplInterface
{
  static const DeviceType D = DeviceType::PrivateUse1;
  CustomNoOpDeviceGuardImpl() = default;
  DeviceType type() const override {
    return D;
  }
  Device exchangeDevice(Device) const override {
    return Device(D, 0); // no-op
  }
  Device getDevice() const override {
    return Device(D, 0);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device) const noexcept override {
    // no-op
  }
  Stream getStream(Device) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getDefaultStream(Device) const override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const override {
    // no-op
    (void)isHighPriority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getNewStream(Device, int priority = 0) const override {
    // no-op
    (void)priority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
  // Event-related functions
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.");
  }
  void block(void* /*event*/, const Stream& /*stream*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
    return true;
  }
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}
  // Stream-related functions
  bool queryStream(const Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomNoOpDeviceGuardImpl);
}
}

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  // NOTE: no idea what this is
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }
};

int register_hook() {
  LOG_DEBUG("register_hook() called");
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());
  LOG_DEBUG("register_hook() SUCCESS - PrivateUse1 hooks registered");
  return 0;
}
int temp_register_hook = register_hook();

at::Tensor wrap_tensor(py::object &py_obj, c10::ScalarType dtype, c10::DeviceIndex device_index) {
  LOG_DEBUG("wrap_tensor() called with dtype=" << (int)dtype << ", device=" << device_index);
  
  std::vector<int64_t> sizes = py_obj.attr("shape").cast<std::vector<int64_t>>();
  
  LOG_DEBUG("  sizes: [" << (sizes.empty() ? std::string("") : std::to_string(sizes[0])) << "]");
  
  // Compute default strides (C-contiguous layout)
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (int i = sizes.size() - 1; i >= 0; i--) {
    strides[i] = stride;
    if (sizes[i] > 1) {
      stride *= sizes[i];
    }
  }
  LOG_DEBUG("  computed strides");

  auto result = at::detail::make_tensor<at::OpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    c10::scalarTypeToTypeMeta(dtype),
    at::Device(at::kPrivateUse1, device_index),
    std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
    sizes);
  
  // Set strides using set_sizes_and_strides
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  
  LOG_DEBUG("  wrap_tensor() SUCCESS - created opaque tensor");
  return result;
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  LOG_DEBUG("unwrap_tensor() called");
  
  auto* impl = tensor.unsafeGetTensorImpl();
  LOG_DEBUG("  got tensor impl");
  
  auto* opaque_impl = static_cast<at::OpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>*>(impl);
  std::shared_ptr<c10::SafePyObject> tiny = opaque_impl->opaque_handle();
  
  LOG_DEBUG("  extracted opaque handle");
  
  auto result = py::reinterpret_borrow<py::object>(tiny->ptr(getPyInterpreter()));
  LOG_DEBUG("  unwrap_tensor() SUCCESS - converted to Python object");
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  LOG_DEBUG("PYBIND11_MODULE initialized");
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
  LOG_DEBUG("Python module functions registered: wrap, unwrap");
}

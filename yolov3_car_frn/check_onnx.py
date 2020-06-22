import onnx


onnx_model = onnx.load("car_frn.onnx")
a = onnx.checker.check_model(onnx_model)
print(a)
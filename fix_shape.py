import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

model_path = "engines/expert_step.int8.qdq.onnx"
data_file = "expert_step.int8.qdq.onnx.data"

print("1. モデルを読み込んでいます...")
model = onnx.load(model_path)

print("2. シンボリックShape推論を実行しています（少し時間がかかります）...")
inferred_model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

print("3. 重みデータを外部ファイルとして保存しています...")
onnx.save_model(
    inferred_model,
    model_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=data_file,
    size_threshold=1024,
    convert_attribute=False
)
print("完了しました！")

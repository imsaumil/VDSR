import torch
import config

def torch2onnx(model_path,dummy_input):
    model = torch.load(model_path,map_location=config.device)
    model.eval()
    output_path = model_path.replace('.torch', '.onnx')
    torch.onnx.export(
            model,
            dummy_input.to(config.device),
            output_path,
            export_params=True,
            do_constant_folding=True,
            input_names=["data"],
            output_names=["output"],
            opset_version=11
        )
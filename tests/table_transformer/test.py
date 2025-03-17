from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
print(type(processor))

processor.save_pretrained("./ckpt/original")
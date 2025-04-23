import torch.utils.tensorboard as tb
import torch
class SummaryWriter:
    def __init__(self, log_dir):
        self.writer = tb.SummaryWriter(log_dir)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int = None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step) 

    def add_histogram(self, tag: str, values: list, global_step: int = None):
        self.writer.add_histogram(tag, values, global_step)

    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: int = None):
        self.writer.add_image(tag, img_tensor, global_step) 

    def add_images(self, tag: str, img_tensor: torch.Tensor, global_step: int = None):
        self.writer.add_images(tag, img_tensor, global_step)

    def add_text(self, tag: str, text_string: str, global_step: int = None):
        self.writer.add_text(tag, text_string, global_step)
    
    def add_audio(self, tag: str, audio_tensor: torch.Tensor, global_step: int = None):
        self.writer.add_audio(tag, audio_tensor, global_step)

    def add_video(self, tag: str, video_tensor: torch.Tensor, global_step: int = None):
        self.writer.add_video(tag, video_tensor, global_step)

    def add_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        self.writer.add_graph(model, input_to_model)

    def add_embedding(self, tag: str, mat: torch.Tensor, metadata: list = None, label_img: torch.Tensor = None, global_step: int =  None):
        self.writer.add_embedding(tag, mat, metadata, label_img, global_step)   

    def add_pr_curve(self, tag: str, labels: list, predictions: list, global_step: int = None):
        self.writer.add_pr_curve(tag, labels, predictions, global_step)
        
        
        

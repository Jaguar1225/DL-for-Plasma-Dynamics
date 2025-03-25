import torch.utils.tensorboard as tb

class SummaryWriter:
    def __init__(self, log_dir):
        self.writer = tb.SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step) 

    def add_histogram(self, tag, values, global_step=None):
        self.writer.add_histogram(tag, values, global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        self.writer.add_image(tag, img_tensor, global_step) 

    def add_images(self, tag, img_tensor, global_step=None):
        self.writer.add_images(tag, img_tensor, global_step)

    def add_text(self, tag, text_string, global_step=None):
        self.writer.add_text(tag, text_string, global_step)
    
    def add_audio(self, tag, audio_tensor, global_step=None):
        self.writer.add_audio(tag, audio_tensor, global_step)

    def add_video(self, tag, video_tensor, global_step=None):
        self.writer.add_video(tag, video_tensor, global_step)

    def add_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def add_embedding(self, tag, mat, metadata=None, label_img=None, global_step=None):
        self.writer.add_embedding(tag, mat, metadata, label_img, global_step)   

    def add_pr_curve(self, tag, labels, predictions, global_step=None):
        self.writer.add_pr_curve(tag, labels, predictions, global_step)
        
        
        

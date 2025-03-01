import importlib
import torch as ch
import os 
import sys
from torchvision.transforms import Normalize
try:
    import timm
except:
    print('Timm models not installed. Please pip install timm if you want to use these models')
from torchvision.models.feature_extraction import create_feature_extractor

def load_metamer_models(model_dir, model_name):
    """
    Use with models from https://github.com/jenellefeather/model_metamers_pytorch. 
    NOTE: Please place this repository on the path. 
    """
    build_network_spec = importlib.util.spec_from_file_location("build_network",
                            os.path.join(model_dir, model_name, 'build_network.py'))
    build_network = importlib.util.module_from_spec(build_network_spec)
    build_network_spec.loader.exec_module(build_network)

    model, ds = build_network.main()
    model.model_name = model_name
    return model, ds

# TODO: If layers are from the same model, we are doing a lot of extra
# computation when things are set up this way. But it is the easiest way to run it...
class WrapMetamerModelLayer(ch.nn.Module):
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.model_name = model.model_name
        if self.layer_name=='final_softmax':
            self.softmax = ch.nn.Softmax()
    
    def forward(self, x, fake_relu=False):
        x = ch.clamp(x, 0, 1) # Include this preprocessing.
        (_, _, all_outputs), _ = self.model(x, with_latent=True)
        if self.layer_name=='final_softmax':
            return self.softmax(all_outputs['final'])
        else:
            return all_outputs[self.layer_name]

# TODO: If layers are from the same model, we are doing a lot of extra
# computation when things are set up this way. But it is the easiest way to run it.
# In the future, could group together model-layer pairs from the same model 
# and extract features from all layers simultaneously to save on compute. 
class WrapTimmModelLayer(ch.nn.Module):
    def __init__(self, model_name, layer_name):
        super().__init__()
        self.model_name = model_name
        self.layer_name = layer_name
        
        # Load the timm model
        temp_model = timm.create_model(model_name, 
                                       pretrained=True, 
                                       )
        temp_model.eval()
        
        # Make preprocessing to include in the gradient computation
        self.mean = temp_model.pretrained_cfg['mean']
        self.std = temp_model.pretrained_cfg['std']
        self.preproc = Normalize(self.mean, self.std)
        
        # Prune the model to only use the layers we are measuring
        return_nodes = {self.layer_name:self.layer_name}
        if 'tf_efficientnet' in self.model_name:
            self.pruned_model = create_feature_extractor(temp_model,
                                                         return_nodes=return_nodes,
                                                         tracer_kwargs={'autowrap_functions': 
                                                                        [timm.layers.padding.pad_same]})
        else:
            self.pruned_model = create_feature_extractor(temp_model, 
                                                         return_nodes=return_nodes)
                
    def forward(self, x):
        x = ch.clamp(x, 0, 1) # Include this preprocessing.
        x = self.preproc(x)
        x = self.pruned_model(x)
        return x[self.layer_name]

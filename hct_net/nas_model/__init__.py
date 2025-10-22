from torch.nn import init
import argparse
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hct_net.nas_model.hybridCnnTransformer import hybridCnnTrans



models_dict={
    'UnetLayer3':hybridCnnTrans,
    'UnetLayer7':hybridCnnTrans,
    'UnetLayer9':hybridCnnTrans,
    'UnetLayer9_v2':hybridCnnTrans,
}


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname!="NoneType":
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            # elif classname.find('BatchNorm2d') != -1:
            #     init.normal_(m.weight.data, 1.0, gain)
            #     init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_models(args,switches_normal,switches_down,switches_up,switches_transformer,early_fix_arch,gen_max_child_flag,random_sample,):

    '''get the correct model '''
    model_name = args.model
    
    # Validate model name exists
    if model_name not in models_dict:
        raise NotImplementedError(f"Model {model_name} does not exist! Available models: {list(models_dict.keys())}")
    
    # Validate layers parameter matches model name (silent auto-correction)
    expected_layers_map = {
        'UnetLayer3': 3,
        'UnetLayer7': 7,
        'UnetLayer9': 9,
        'UnetLayer9_v2': 9,
    }
    
    if model_name in expected_layers_map:
        expected_layers = expected_layers_map[model_name]
        if args.layers != expected_layers:
            # Silently adjust layers to match model name expectation
            args.layers = expected_layers
    
    # Create model with unified parameters
    model = models_dict[model_name](
        input_c=args.input_c,
        c=args.init_channel,
        num_classes=args.num_classes,
        meta_node_num=args.meta_node_num,
        layers=args.layers,
        dp=args.dropout_prob,
        use_sharing=args.use_sharing,
        double_down_channel=args.double_down_channel,
        use_softmax_head=args.use_softmax_head,
        switches_normal=switches_normal,
        switches_down=switches_down,
        switches_up=switches_up,
        early_fix_arch=args.early_fix_arch,
        gen_max_child_flag=args.gen_max_child_flag,
        random_sample=args.random_sample
    )
    
    init_weights(model, args.init_weight_type)
    return model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_class=1
    args.im_ch=3
    args.init_channel=16
    args.middle_nodes=4
    args.layers=7
    args.init_weight_type="kaiming"
    args.model="UnetLayer7"
    model=get_models(args)
    print(model)
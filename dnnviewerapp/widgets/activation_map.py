from dnnviewerapp.layers import AbstractLayer, Convo2D
from dnnviewerapp.bridge import AbstractActivationMapper
import dnnviewerapp.imageutils as imageutils

import dash_html_components as html


def widget(activation_mapper: AbstractActivationMapper, layer: AbstractLayer, unit_idx: int, input_img):
    """ Activation maps widget """

    if isinstance(layer, Convo2D):
        maps = activation_mapper.get_activation(input_img, layer, unit_idx)
        if unit_idx is None:
            return [html.Div(html.Img(id='activation-map', alt='Activation map',
                                      src=imageutils.array_to_img_src(imageutils.to_8bit_img(img))),
                             className='thumbnail') for img in maps]
        else:
            return [html.H5('Unit #%s activation' % unit_idx),
                    html.Div(html.Img(id='activation-map', alt='Activation map',
                                      src=imageutils.array_to_img_src(imageutils.to_8bit_img(maps))),
                             className='thumbnail')]

import torch
import soft_renderer as sr


class NeuralRenderer(torch.nn.Module):
    def __init__(self, img_size=256, camera_mode='look_at', orig_size=256, background=[0, 0, 0], 
                 texture_type='surface', anti_aliasing=False, **kwargs):
        super(NeuralRenderer, self).__init__()
        self.camera_mode = camera_mode
        self.texture_type = texture_type

        self.renderer = sr.SoftRenderer(image_size=img_size, camera_mode=camera_mode, orig_size=orig_size,
                                        background_color=background, texture_type=self.texture_type,
                                        anti_aliasing=anti_aliasing, **kwargs)
        self.renderer = self.renderer.cuda()

    def set_camera(self, K, Rt):
        # set 3x4 projection matrix (K @ Rt)
        P = K @ Rt
        self.renderer.set_transform(P)

    def set_lighting(self, light_intensity_ambient=0.5, light_intensity_directional=0.5, light_direction=[0, 1, 0]):
        # set renderer light values
        self.renderer.set_lighting(light_intensity_ambient, light_intensity_directional, light_direction)

    def forward(self, vertices, faces, textures=None, mode=None):
        vs = vertices.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        if textures is None:
            ts = textures
        else:
            ts = textures.clone()

        imgs = self.renderer(vs, fs, ts)
        imgs = torch.flip(imgs, (2,))  # invert y axis

        text, mask = imgs[:, :-1], imgs[:, -1]

        return text, mask

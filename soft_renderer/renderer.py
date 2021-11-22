
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

import soft_renderer as sr


class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=True, fill_back=True, eps=1e-6,
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(Renderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.Rasterizer(image_size, background_color, near, far, 
                                        anti_aliasing, fill_back, eps)

    def forward(self, mesh, mode=None):
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)


class SoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0], clamp_lighting=False):
        super(SoftRenderer, self).__init__()

        self.light_mode = light_mode
        self.light_color_ambient = light_color_ambient
        self.light_color_directionals = light_color_directionals
        self.clamp_lighting = clamp_lighting

        self.default_texture_type = texture_type

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions, clamp_lighting=self.clamp_lighting)

        if P is None:
            P = torch.eye(4)[:-1, :].unsqueeze(0).cuda()

        self.camera_mode = camera_mode
        self.dist_coeffs = dist_coeffs
        self.orig_size = orig_size
        if self.camera_mode != 'projection':
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.viewing_scale = viewing_scale
            self.eye = eye
            self.camera_direction = camera_direction

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)

    def set_lighting(self, light_intensity_ambient, light_intensity_directionals, light_directions):
        self.lighting = sr.Lighting(self.light_mode, light_intensity_ambient, self.light_color_ambient,
                                    light_intensity_directionals, self.light_color_directionals, light_directions,
                                    clamp_lighting=self.clamp_lighting)

    def set_transform(self, P, dist_coeffs=None):
        self.transform = sr.Transform(self.camera_mode, P, dist_coeffs, self.orig_size)

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh, mode=None):
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)

    def forward(self, vertices, faces, textures=None, mode=None, texture_type=None):
        if texture_type is None:
            texture_type = self.default_texture_type
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        return self.render_mesh(mesh, mode)
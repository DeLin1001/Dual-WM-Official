from collections.abc import Sequence
import numpy as np
import pygame
import pymunk
import shapely.geometry as sg
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
positive_y_is_up: bool = False

class DrawOptions(pymunk.SpaceDebugDrawOptions):

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        super().__init__()

    def draw_circle(self, pos: Vec2d, angle: float, radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(self, a: tuple[float, float], b: tuple[float, float], radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [(p1[0] - orthog[0], p1[1] - orthog[1]), (p1[0] + orthog[0], p1[1] + orthog[1]), (p2[0] + orthog[0], p2[1] + orthog[1]), (p2[0] - orthog[0], p2[1] - orthog[1])]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(self.surface, fill_color.as_int(), (round(p1[0]), round(p1[1])), round(radius))
            pygame.draw.circle(self.surface, fill_color.as_int(), (round(p2[0]), round(p2[1])), round(radius))

    def draw_polygon(self, verts: Sequence[tuple[float, float]], radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]
        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)
        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(self, size: float, pos: tuple[float, float], color: SpaceDebugColor) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)

def get_mouse_pos(surface: pygame.Surface) -> tuple[int, int]:
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)

def to_pygame(p: tuple[float, float], surface: pygame.Surface) -> tuple[int, int]:
    if positive_y_is_up:
        return (round(p[0]), surface.get_height() - round(p[1]))
    else:
        return (round(p[0]), round(p[1]))

def from_pygame(p: tuple[float, float], surface: pygame.Surface) -> tuple[int, int]:
    return to_pygame(p, surface)

def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        elif isinstance(shape, pymunk.shapes.Circle):
            center = body.local_to_world(shape.offset)
            poly = sg.Point(tuple(center)).buffer(shape.radius, resolution=16)
            geoms.append(poly)
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

def perturb_camera_angle(xyaxis, deg_dif=[3, 3]):
    xaxis = np.array(xyaxis[:3])
    yaxis = np.array(xyaxis[3:])
    zaxis = np.cross(xaxis, yaxis)
    zaxis /= np.linalg.norm(zaxis)
    yaw = np.deg2rad(deg_dif[0])
    pitch = np.deg2rad(deg_dif[1])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R = R_pitch @ R_yaw
    xaxis_new = R @ xaxis
    yaxis_new = R @ yaxis
    xyaxes_new = tuple(np.concatenate([xaxis_new, yaxis_new]))
    return xyaxes_new

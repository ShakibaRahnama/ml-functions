"""
Set of functions to draw segmentations exported from Encord.
"""
import re
import numpy as np
import cv2
import math

# valid_shapes has keys which are the shape_types, and values which is the relevant key for the passed in object needed to get the coordinate information to be passed into the draw method.
valid_shapes = {'polygon': 'polygon',
                'rotatable_bounding_box': 'rotatableBoundingBox',
                'bounding_box': 'boundingBox'}
class EncordObject():
    """
    Formal object for extracted segmentation information from a json file.
    """
    def __init__(self, object):
        """
        Extract all objects from the list of objects as recorded in an Encord
        json file.
        Args:
            objects: dict, dictionary with specific set of keys required. See
                method valid_object for more details.
        """
        val, why = self.valid_object(object)
        if not val:
            err_msg = f"Problem with object: {why}"
            raise ValueError(err_msg)
        self.name = object['name']
        self.color = object['color']
        self.shape_type = object['shape']
        try:
            self.shape = object[valid_shapes[object['shape']]]
        except KeyError:
            msg = (f"{self.shape_type} which has expected key "
                   f"{valid_shapes[object['shape']]} was not found.")
            raise KeyError(msg)
                
    def valid_object(self, obj):
        if not 'name' in obj.keys():
            return False, "Missing key name"
        if not isinstance(obj['name'], str):
            return False, "Name value is not a string"
        if not 'color' in obj.keys():
            return False, "Missing key color"
        color_pat = re.compile(r"^#(?:[A-Fa-f0-9]{3}){1,2}$")
        if not color_pat.match(obj['color']):
            return False, "Color not a valid CSS color"
        if not 'shape' in obj.keys():
            return False, "Missing key shape"
        if not obj['shape'] in valid_shapes:
            return False, (f"obj['shape'] {obj['shape']}, missing from "
                           f"valid_shapes list.")
        # TODO: are there ways to check the validity of the actual given shape?
        if not 'value' in obj.keys():
            return False, "Missing key value"
        if not isinstance(obj['value'], str):
            return False, "obj['value'] is not a string"

        return True, ''
    
    def draw(self, bg: np.array, color: tuple|None = None, alpha=1):
        """
        Draw the segmentation object on the given image bg.

        Args:
            bg: np.array, a numpy array of an image, that the segmentation
                should be drawn on.
            color: tuple, of three integers or None. If None then the color
                stored in self will be used. If not None, the three integers
                must be values in [0,255] and will be interpreted as a BGR
                color.
            alpha: float, between 0-1. alpha channel value for the color if
                the color in the EncordObject is used, if color is given then
                assume it has an alpha channel included.
        Returns:
            np.array, bg with the segmentation drawn on top.
        """
        if self.shape_type == 'polygon':
            return self._draw_polygon(bg, color, alpha)
        elif self.shape_type == 'rotatable_bounding_box':
            return self._draw_rotatable_bb(bg, color, alpha)
        elif self.shape_type == 'bounding_box':
            return self._draw_bb(bg, color, alpha)
        else:
            msg = (f"Tried to draw a shape type {self.shape_type} that is not "
                   f"implemented")
            raise NotImplementedError()

    def _draw_polygon(self, bg: np.array, color: tuple|None = None, alpha=1):
        """
        Helper function for drawing the polygon shape types
        """
        coord_array = []
        for coords in self.shape.values():
            coord_array.append((int(coords['x']*bg.shape[1]),
                                int(coords['y']*bg.shape[0])))
        coord_array = np.array(coord_array, dtype=np.int32).reshape((-1,1,2))
        if not color:
            BGRcolor = (int(self.color[5:7], 16), int(self.color[3:5], 16),
                        int(self.color[1:3], 16))
        else:
            BGRcolor = color
        drawn_obj = cv2.fillPoly(bg.copy(), [coord_array], color=BGRcolor)
        combined_img = cv2.addWeighted(src1=drawn_obj, alpha=alpha, src2=bg,
                                       beta=1 - alpha, gamma=0)
        return combined_img

    def _draw_rotatable_bb(self, bg: np.array, color: tuple|None = None,
                           alpha=1):
        """
        Helper function for drawing the rotatable_bounding_box shape types
        """
        # keep these values floats and only round to ints after rotation
        width = self.shape['w']*bg.shape[1]
        height = self.shape['h']*bg.shape[0]
        theta = self.shape['theta']

        # keep all values as floats first then convert to int pixel locations prior to drawing.
        p0 = (self.shape['x']*bg.shape[1],
              self.shape['y']*bg.shape[0])
        p1 = (p0[0] + (width*math.cos(theta)),
              p0[1] + (width*math.sin(theta)))
        p2 = (p1[0] + (height*math.cos(theta - (math.pi/2))),
              p1[0] + (height*math.sin(theta - (math.pi/2))))
        p3 = (p2[0] + (width*math.cos(theta - math.pi)),
              p2[1] + width*math.sin(theta - math.pi))
        
        coord_array = np.array([p0, p1, p2, p3],
                               dtype=np.int32).reshape((-1,1,2))

        if not color:
            BGRcolor = (int(self.color[5:7], 16), int(self.color[3:5], 16),
                        int(self.color[1:3], 16))
        else:
            BGRcolor = color
        
        drawn_obj = cv2.fillPoly(bg.copy(), [coord_array], color=BGRcolor)
        combined_img = cv2.addWeighted(src1=drawn_obj, alpha=alpha, src2=bg,
                                       beta=1 - alpha, gamma=0)

        return combined_img

    def _draw_bb(self, bg: np.array, color: tuple|None = None, alpha=1):
        """
        Helper function for drawing the bounding_box shape types
        """
        # keep these values floats and only round to ints after rotation
        width = self.shape['w']*bg.shape[1]
        height = self.shape['h']*bg.shape[0]
        theta = 0

        # keep all values as floats first then convert to int pixel locations prior to drawing.
        p0 = (self.shape['x']*bg.shape[1],
              self.shape['y']*bg.shape[0])
        p1 = (p0[0] + (width*math.cos(theta)),
              p0[1] + (width*math.sin(theta)))
        p2 = (p1[0] + (height*math.cos(theta - (math.pi/2))),
              p1[0] + (height*math.sin(theta - (math.pi/2))))
        p3 = (p2[0] + (width*math.cos(theta - math.pi)),
              p2[1] + width*math.sin(theta - math.pi))
        
        coord_array = np.array([p0, p1, p2, p3],
                               dtype=np.int32).reshape((-1,1,2))

        if not color:
            BGRcolor = (int(self.color[5:7], 16), int(self.color[3:5], 16),
                        int(self.color[1:3], 16))
        else:
            BGRcolor = color
        
        drawn_obj = cv2.fillPoly(bg.copy(), [coord_array], color=BGRcolor)
        combined_img = cv2.addWeighted(src1=drawn_obj, alpha=alpha, src2=bg,
                                       beta=1 - alpha, gamma=0)

        return combined_img

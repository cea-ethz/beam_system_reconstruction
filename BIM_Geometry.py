import numpy as np
import open3d as o3d


class Geometry:
    geometry_id_counter = 0

    def __init__(self):
        self.id = Geometry.geometry_id_counter
        Geometry.geometry_id_counter += 1


class Collision:
    def __init__(self, a, b, precedence):
        self.a = a
        self.b = b
        self.precedence = precedence

    def equals(self, other):
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)


class BeamSystemLayer:
    def __init__(self):
        self.beams = []
        self.mean_spacing = -1
        self.average_z = None

    def add_beam(self, beam):
        self.beams.append(beam)

    def finalize(self):
        if len(self.beams):
            distances = np.zeros(len(self.beams) - 1)
            for i in range(len(self.beams) - 1):
                beam_a = self.beams[i]
                beam_b = self.beams[i + 1]
                distances[i] = abs(beam_a.aabb.get_center()[beam_a.axis] - beam_b.aabb.get_center()[beam_b.axis])
                self.mean_spacing = np.mean(distances)

        self.average_z = np.average([beam.aabb.get_center()[2] for beam in self.beams])


class Beam(Geometry):

    def __init__(self, aabb, axis, cloud):
        super().__init__()
        self.aabb = aabb
        self.axis = axis
        self.long_axis = int(not axis)
        self.cloud = cloud
        self.length = aabb.get_extent()[int(not axis)]

    def check_overlap(self, other):
        sc = self.aabb.get_center()
        se = self.aabb.get_half_extent()
        oc = other.aabb.get_center()
        oe = other.aabb.get_half_extent()

        x = abs(sc[0] - oc[0]) < (se[0] + oe[0])
        y = abs(sc[1] - oc[1]) < (se[1] + oe[1])
        z = abs(sc[2] - oc[2]) < (se[2] + oe[2])

        # -1 Precedence implies a corner connection
        precedence = -1

        # Beams sitting on top of each other
        if (sc[2] - oc[2]) > (se[2] + oe[2]) * 0.75:
            precedence = self.axis

        if (oc[2] - sc[2]) > (oe[2] + se[2]) * 0.75:
            precedence = other.axis

        # Beams touching
        # Except first we need beam splitting

        if x and y and z:
            return Collision(self, other, precedence)
        else:
            return None

    def get_point_param(self, point):
        """Returns the world units position of a point projected to the beam"""
        p_value = point[self.long_axis]
        p_value -= (self.aabb.get_center()[self.long_axis] - self.aabb.get_half_extent()[self.long_axis])
        return p_value

    def split(self, position):
        min_bound = self.aabb.get_min_bound()
        max_bound = self.aabb.get_max_bound()

        max_a = np.copy(max_bound)
        max_a[self.long_axis] = position + min_bound[self.long_axis]

        min_b = np.copy(min_bound)
        min_b[self.long_axis] = position + min_bound[self.long_axis]

        aabb_a = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_a)
        aabb_b = o3d.geometry.AxisAlignedBoundingBox(min_b, max_bound)

        if self.cloud is not None:
            pc_a = self.cloud.crop(aabb_a)
            pc_b = self.cloud.crop(aabb_b)

            beam_a = Beam(aabb_a, self.axis, pc_a)
            beam_b = Beam(aabb_b, self.axis, pc_b)

        else:
            beam_a = Beam(aabb_a, self.axis, None)
            beam_b = Beam(aabb_b, self.axis, None)



        return beam_a, beam_b


class Column(Geometry):
    def __init__(self, center):
        super().__init__()
        self.center = center
        self.child_beams = []
        self.cloud = None
        self.aabb = None

import numpy as np
from .forwardKinematics import ForwardKinematic
import plotly.graph_objects as go
from urbasic import ISCoin, Joint6D

class RobotCollisionCheck:
    def __init__(self, angles: list[float]):
        self.angles = angles
        self.diameters = {
            1: 0.4,  # to be defined
            2: 0.128,
            3: 0.08,
            4: 0.065,
            5: 0.085,
            6: 0.085,
        }
        self.safeDistances = {
            1: {2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05}, # real values (1,5) 0.1
            2: {3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05},
            3: {4: 0.05, 5: 0.05, 6: 0.05},# real 2,4 0.3
            4: {5: 0.05, 6: 0.05},
            5: {6: 0.05},
        }
        self.cylinders = {}
        self.coordinates = ForwardKinematic(angles).getCoordinates()
        self.safeDistancesFromTheGround = 0.02 # safe distance from the ground
        self._fillCylindersDict()

    def _createVectorCylinder(self, p1, q1):
        """Creation of a normalized vector from two points
        to define the direction of the cylinder"""
        vector = np.array(
            [(q1["x"] - p1["x"]), (q1["y"] - p1["y"]), (q1["z"] - p1["z"])]
        )
        return vector

    def _createCylinder(self, key, p1, q1):
        """Create a cylinder with the given coordinates and radius"""
        if key == 5:
            # Extend the cylinder with key 6 by scaling its direction vector
            extension_factor = 1.2
            direction_vector = self._createVectorCylinder(p1, q1)
            extended_p1 = {
                "x": p1["x"] - extension_factor * direction_vector[0],
                "y": p1["y"] - extension_factor * direction_vector[1],
                "z": p1["z"] - extension_factor * direction_vector[2],
            }
            
            extended_q1 = {
                "x": q1["x"] + extension_factor * direction_vector[0],
                "y": q1["y"] + extension_factor * direction_vector[1],
                "z": q1["z"] + extension_factor * direction_vector[2],
            }
            self.cylinders[key] = {
                "p": extended_p1,
                "q": extended_q1,
                "r": self.diameters[key] / 2,
                "d": direction_vector,
            }
        else:
            extension_factor = 0.1
            direction_vector = self._createVectorCylinder(p1, q1)
            # Extend both p1 and q1 along the direction vector
            extended_p1 = {
                "x": p1["x"] - extension_factor * direction_vector[0],
                "y": p1["y"] - extension_factor * direction_vector[1],
                "z": p1["z"] - extension_factor * direction_vector[2],
            }
            
            extended_q1 = {
                "x": q1["x"] + extension_factor * direction_vector[0],
                "y": q1["y"] + extension_factor * direction_vector[1],
                "z": q1["z"] + extension_factor * direction_vector[2],
            }
            
            # Store the cylinder with extended points
            self.cylinders[key] = {
                "p": extended_p1,
                "q": extended_q1,
                "r": self.diameters[key] / 2,
                "d": direction_vector,
            }

    def _fillCylindersDict(self):
        for key, value in self.coordinates.items():
            if key < 6:
                self._createCylinder(key, value, self.coordinates[key + 1])
            else:
                break

    def _closest_points_between_segments(self, p1, q1, p2, q2):
        """Robust method to find closest points between two line segments."""
        # Convert points to numpy arrays
        p1 = np.array([p1["x"], p1["y"], p1["z"]])
        q1 = np.array([q1["x"], q1["y"], q1["z"]])
        p2 = np.array([p2["x"], p2["y"], p2["z"]])
        q2 = np.array([q2["x"], q2["y"], q2["z"]])
        
        # Direction vectors
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        f = np.dot(r, r)
        
        denom = a*c - b*b
        
        # Default values
        s = 0.0
        t = 0.0
        
        if denom < 1e-6:  # Segments are nearly parallel
            # Handle parallel segments properly
            s = 0.0
            if b > c:  # Project p1 onto d2
                t = d / b
            else:       # Project p2 onto d1
                t = e / c
            t = np.clip(t, 0, 1)
        else:
            # General case
            s = (b*e - c*d) / denom
            t = (a*e - b*d) / denom
            
            # Check if closest points are outside segments
            # and need to be clamped to endpoints
            if s < 0:
                s = 0
                t = e / c
                t = np.clip(t, 0, 1)
            elif s > 1:
                s = 1
                t = (e + b) / c
                t = np.clip(t, 0, 1)
            
            if t < 0:
                t = 0
                s = -d / a
                s = np.clip(s, 0, 1)
            elif t > 1:
                t = 1
                s = (b - d) / a
                s = np.clip(s, 0, 1)
        
        # Final closest points
        closest_p1 = p1 + s * d1
        closest_p2 = p2 + t * d2
        
        return closest_p1, closest_p2

    def _computeDistanceBetweenTwoCylinders(self, cylinderKey1, cylinderKey2):
        """Compute the minimum distance between two cylinders considering their radii."""
        # Get cylinder parameters
        p1, q1, r1 = self.cylinders[cylinderKey1]["p"], self.cylinders[cylinderKey1]["q"], self.cylinders[cylinderKey1]["r"]
        p2, q2, r2 = self.cylinders[cylinderKey2]["p"], self.cylinders[cylinderKey2]["q"], self.cylinders[cylinderKey2]["r"]
        
        # Convert to numpy arrays
        p1 = np.array([p1["x"], p1["y"], p1["z"]])
        q1 = np.array([q1["x"], q1["y"], q1["z"]])
        p2 = np.array([p2["x"], p2["y"], p2["z"]])
        q2 = np.array([q2["x"], q2["y"], q2["z"]])
        
        # Direction vectors and lengths
        u = q1 - p1
        v = q2 - p2
        len_u = np.linalg.norm(u)
        len_v = np.linalg.norm(v)
        u_norm = u / len_u if len_u > 0 else u
        v_norm = v / len_v if len_v > 0 else v
        
        # Find closest points between centerlines
        closest_p1, closest_p2 = self._closest_points_between_segments(
            {"x": p1[0], "y": p1[1], "z": p1[2]},
            {"x": q1[0], "y": q1[1], "z": q1[2]},
            {"x": p2[0], "y": p2[1], "z": p2[2]},
            {"x": q2[0], "y": q2[1], "z": q2[2]}
        )
        
        # Calculate parameter values (s,t) along each segment
        s = np.dot(closest_p1 - p1, u) / (len_u**2) if len_u > 0 else 0
        t = np.dot(closest_p2 - p2, v) / (len_v**2) if len_v > 0 else 0
        
        # Determine which parts of the cylinders are closest
        on_side1 = (1e-6 < s < 1-1e-6)  # Point is on side of cylinder 1
        on_side2 = (1e-6 < t < 1-1e-6)  # Point is on side of cylinder 2
        distance = 0.0
        if on_side1 and on_side2:
            # Both closest points are on cylinder sides
            cos_theta = abs(np.dot(u_norm, v_norm)) if len_u > 0 and len_v > 0 else 0
            r_effective = np.sqrt(r1**2 + r2**2 - 2*r1*r2*cos_theta)
            distance = np.linalg.norm(closest_p2 - closest_p1) - r_effective
        elif on_side1:
            # Closest point on cylinder 2 is an endcap
            distance = np.linalg.norm(closest_p2 - closest_p1) - r1
        elif on_side2:
            # Closest point on cylinder 1 is an endcap
            distance = np.linalg.norm(closest_p2 - closest_p1) - r2
        else:
            # Both closest points are endcaps
            distance = np.linalg.norm(closest_p2 - closest_p1) - 0  # No radius subtraction for endcap-endcap
        
        return max(distance, 0)  # Return 0 if cylinders intersect

    def _followTheLine(self, p1, q1, p2, q2):
        closest_p1, closest_p2 = self._closest_points_between_segments(
            p1, q1, p2, q2
        )
        points = []
        steps = 0.1
        for t in np.arange(0, 1, steps):
            point = (1 - t) * closest_p1 + t * closest_p2
            points.append(point)
        return points

    def point_in_cylinder(self, point, point1, point2, radius):
        """
        Check if a point is inside a cylinder defined by two points and a radius.
        
        Args:
        - point: Point to check
        - point1: First point defining the cylinder axis
        - point2: Second point defining the cylinder axis
        - radius: Radius of the cylinder
        
        Returns:
        Tuple of (is_inside, point)
        """
        # Convert to numpy arrays
        point = np.array(point)
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # Vector along cylinder axis
        axis = point2 - point1
        axis_length = np.linalg.norm(axis)
        if axis_length < 1e-6:  # Points are coincident
            distance = np.linalg.norm(point - point1)
            return (distance <= radius, point)
            
        axis_normalized = axis / axis_length
        
        # Vector from point to cylinder base
        point_vector = point - point1
        
        # Projection on cylinder axis
        projection = np.dot(point_vector, axis_normalized)
        
        # Check if projection is within cylinder length
        if projection < 0 or projection > axis_length:
            return (False, point)
        
        # Point projected on axis
        projected_point = point1 + projection * axis_normalized
        
        # Distance from point to axis
        distance_to_axis = np.linalg.norm(point - projected_point)
        
        return (distance_to_axis <= radius, point)

    def getFirstPoint(self, cylinderKey1, cylinderKey2):
        """Get the first point where the line exits cylinder1"""
        points = self._followTheLine(
            self.cylinders[cylinderKey1]["p"],
            self.cylinders[cylinderKey1]["q"],  
            self.cylinders[cylinderKey2]["p"],
            self.cylinders[cylinderKey2]["q"],
        )
        
        p1 = np.array([self.cylinders[cylinderKey1]["p"][key] for key in ["x", "y", "z"]])
        q1 = np.array([self.cylinders[cylinderKey1]["q"][key] for key in ["x", "y", "z"]])
        radius = self.cylinders[cylinderKey1]["r"]
        
        for point in points:
            is_inside, _ = self.point_in_cylinder(point, p1, q1, radius)
            
            if not is_inside:
                print(f"Point: {point}, cylinderk : {cylinderKey1}Inside: {is_inside}")
                return point
                
        # If no point found, return the last point of cylinder1
        return np.array([self.cylinders[cylinderKey1]["q"][key] for key in ["x", "y", "z"]])

    def getSecondPoint(self, cylinderKey1, cylinderKey2):
        """Get the first point where the line enters cylinder2"""
        points = self._followTheLine(
            self.cylinders[cylinderKey1]["p"],
            self.cylinders[cylinderKey1]["q"],  
            self.cylinders[cylinderKey2]["p"],
            self.cylinders[cylinderKey2]["q"],
        )
        
        p2 = np.array([self.cylinders[cylinderKey2]["p"][key] for key in ["x", "y", "z"]])
        q2 = np.array([self.cylinders[cylinderKey2]["q"][key] for key in ["x", "y", "z"]])
        radius = self.cylinders[cylinderKey2]["r"]
        
        for point in reversed(points):  # Check from the end
            is_inside, _ = self.point_in_cylinder(point, p2, q2, radius)
            if is_inside:
                print(f"Point: {point}, cylinderk : {cylinderKey2}Inside: {is_inside}")

                return point
                
        # If no point found, return the first point of cylinder2
        return np.array([self.cylinders[cylinderKey2]["p"][key] for key in ["x", "y", "z"]])

    def getDistanceWithRadii(self, cylinderKey1, cylinderKey2):
        """Get the distance between two cylinders considering their radii"""
        newP1 = self.getFirstPoint(cylinderKey1, cylinderKey2)
        newP2 = self.getSecondPoint(cylinderKey1, cylinderKey2)
        
            # Fallback to centerline distance if points weren't found
        p1 = np.array([self.cylinders[cylinderKey1]["p"][key] for key in ["x", "y", "z"]])
        q1 = np.array([self.cylinders[cylinderKey1]["q"][key] for key in ["x", "y", "z"]])
        p2 = np.array([self.cylinders[cylinderKey2]["p"][key] for key in ["x", "y", "z"]])
        q2 = np.array([self.cylinders[cylinderKey2]["q"][key] for key in ["x", "y", "z"]])
        
        closest_p1, closest_p2 = self._closest_points_between_segments(
            {"x": p1[0], "y": p1[1], "z": p1[2]},
            {"x": q1[0], "y": q1[1], "z": q1[2]},
            {"x": p2[0], "y": p2[1], "z": p2[2]},
            {"x": q2[0], "y": q2[1], "z": q2[2]}
        )
        
        return np.sqrt(
            np.sum((newP1 - newP2) ** 2)
        ), newP1, newP2

    def checkingCollisionWithItself(self):
        cylinderDistances = {}
        cylinder_keys = list(self.cylinders.keys())

        for i, key1 in enumerate(cylinder_keys):
            for key2 in cylinder_keys[i + 1:]:
                if key2 == key1 + 1:  # Skip consecutive cylinders
                    continue

                if key1 in self.safeDistances and key2 in self.safeDistances[key1]:
                    distance,newP1,newP2 = self.getDistanceWithRadii(key1, key2)
                    safe_distance = self.safeDistances[key1][key2]
                    cylinderDistances[(key1, key2)] = distance > safe_distance
                    
        return cylinderDistances

    def checkingCollisionWithGround(self):
        result = {}
        for joint_index in range(1, 7):  # Check all 6 joints
            if joint_index in self.cylinders:
                coord = self.cylinders[joint_index]
                result[joint_index] = coord["q"]["z"] - self.safeDistancesFromTheGround >= 0.0
        return result

    def plotCylinders(self):
        """Plot the cylinders in 3D space interactively using plotly"""
        fig = go.Figure()
        
        # Plot cylinders
        for key, cylinder in self.cylinders.items():
            p = np.array([cylinder["p"]["x"], cylinder["p"]["y"], cylinder["p"]["z"]])
            q = np.array([cylinder["q"]["x"], cylinder["q"]["y"], cylinder["q"]["z"]])
            r = cylinder["r"]
            
            # Create cylinder mesh
            cylinder_mesh = self._create_cylinder_mesh(p, q, r)
            fig.add_trace(cylinder_mesh)
            
            # Add label
            midpoint = (p + q) / 2
            fig.add_trace(go.Scatter3d(
                x=[midpoint[0]],
                y=[midpoint[1]],
                z=[midpoint[2]],
                text=[f"Cylinder {key}"],
                mode="text",
                showlegend=False
            ))
            
        # Plot distances between cylinders that are checked
        collision_results = self.checkingCollisionWithItself()
        for (key1, key2), is_safe in collision_results.items():
         
            cyl1 = self.cylinders[key1]
            cyl2 = self.cylinders[key2]
            
            dist, closest_p1, closest_p2 = self.getDistanceWithRadii(key1, key2)
            
            # Plot the line connecting the closest points
            fig.add_trace(go.Scatter3d(
            x=[closest_p1[0], closest_p2[0]],
            y=[closest_p1[1], closest_p2[1]],
            z=[closest_p1[2], closest_p2[2]],
            mode="lines",
            line=dict(color="red", width=4),
            name=f"Collision {key1}-{key2}",
            ))
            
            # Plot the closest points
            fig.add_trace(go.Scatter3d(
            x=[closest_p1[0]],
            y=[closest_p1[1]],
            z=[closest_p1[2]],
            mode="markers",
            marker=dict(color="green", size=6),
            name=f"Closest Point {key1}",
            ))
            fig.add_trace(go.Scatter3d(
            x=[closest_p2[0]],
            y=[closest_p2[1]],
            z=[closest_p2[2]],
            mode="markers",
            marker=dict(color="blue", size=6),
            name=f"Closest Point {key2}",
            ))

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data"
            ),
            title="Robot Collision Visualization",
            showlegend=True
        )
        
        fig.show()

    def _create_cylinder_mesh(self, p, q, radius, color="blue", opacity=0.5):
        """Helper function to create a cylinder mesh for plotting"""
        # Vector along cylinder axis
        vec = q - p
        vec_length = np.linalg.norm(vec)
        
        # Create a unit cylinder along z-axis
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, vec_length, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        
        # Rotate and translate the cylinder
        # First find rotation to align with vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, vec)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_length
            rotation_angle = np.arccos(np.dot(z_axis, vec) / vec_length)
            
            # Apply rotation
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_rotvec(rotation_angle * rotation_axis)
            xyz = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
            xyz_rotated = rot.apply(xyz)
            
            x_grid = xyz_rotated[:, 0].reshape(x_grid.shape)
            y_grid = xyz_rotated[:, 1].reshape(y_grid.shape)
            z_grid = xyz_rotated[:, 2].reshape(z_grid.shape)
        
        # Translate to position p
        x_grid += p[0]
        y_grid += p[1]
        z_grid += p[2]
        
        # Create mesh
        return go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            name=f"Cylinder r={radius:.3f}"
        )


if __name__ == "__main__":
    testAngles = [
        -0.1122,
        -3.0658,
        0.4916,
        2.5814,
        1.4586,
        -1.5638
    ]
    test = RobotCollisionCheck(testAngles)
    print("Collision check results:", test.checkingCollisionWithItself())
    print("Ground collision check:", test.checkingCollisionWithGround())
    test.plotCylinders()
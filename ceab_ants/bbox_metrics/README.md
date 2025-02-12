
BBOX in a **left**, **top**, **right**, **bottom** format \
BBOX observations in a **center x**, **center y**, **size**, **aspect ratio** format

OBBOX in a **center x**, **center y**, **width**, **height**, **angle** format \
OBBOX observations in a **center x**, **center y**, **size**, **aspect ratio**, **angle** format

# OBBOX intersections
* 8 intersections + 0 inside vertices will form a convex irregular octagon
* 6 intersections + 1 inside vertex  will form a convex irregular heptagon
* 4 intersections + 2 inside vertices will form a convex irregular hexagon
* 2 intersections + 3 inside vertices will form a convex irregular pentagon
* 2 intersections + 2 inside vertices will form a convex quadrilater
* 2 intersections + 1 inside vertex will form a triangle
* 0 intersections + 4 inside vertices will form the original obbox that's inside the other
* Invalid cases have 0 (separated), 1 (point) or 2 (line) valid points
* The output array shape should be (N, M, 8, 2) with one point repited "8 - num_intersections - num_vertices + 1" times and 8 np.nan if no intersection

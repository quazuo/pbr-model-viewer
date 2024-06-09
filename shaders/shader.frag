#version 450

#include "ext.glsl"
#include "utils.glsl"

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
};

struct ColoringData {
    uint do_neighbor_shading;
    uint preset_id;
    vec3 cell_color1;
    vec3 cell_color2;
    vec3 background_color;
};

struct MiscData {
    float fog_dist;
    vec3 camera_pos;
};

struct AutomatonInfo {
    uint grid_depth;
    uint state_count;
};

layout(set = 0, binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    ColoringData coloring;
    MiscData misc;
    AutomatonInfo automaton;
} ubo;

layout(set = 1, binding = 1) readonly buffer StateSsboIn {
    uint8_t state_in[];
};

struct Node {
    uint index;
    uint8_t value;
};

struct Octree {
    float size;
    vec3 min;
    vec3 max;
    Node root;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct RayResult {
    Node node;
    float dist;
    vec3 normal;
};

#define NULL_RESULT_DIST -1.0
#define NULL_RESULT RayResult(Node(0, uint8_t(0)), NULL_RESULT_DIST, vec3(0))

bool is_null_result(RayResult res) {
    return res.dist == NULL_RESULT_DIST;
}

uint8_t read_value(uint index, uint level) {
    return state_in[get_level_offset(level) + index];
}

Node get_child(Node parent, uint index, uint level) {
    Node child;
    child.index = 8 * parent.index + index;
    child.value = read_value(child.index, level + 1);

    return child;
}

uint first_node(float tx0, float ty0, float tz0, float txm, float tym, float tzm) {
    uint result = 0;
    const float tmax = max(tx0, ty0, tz0);

    if (tmax == tx0) {
        if (tym < tx0) result |= 2u;
        if (tzm < tx0) result |= 1u;
    } else if (tmax == ty0) {
        if (txm < ty0) result |= 4u;
        if (tzm < ty0) result |= 1u;
    } else { // if (tmax == tz0)
        if (txm < tz0) result |= 4u;
        if (tym < tz0) result |= 2u;
    }

    return result;
}

uint new_node(float tx, uint nx, float ty, uint ny, float tz, uint nz) {
    float tmin = min(tx, ty, tz);
    if (tmin == tx) return nx;
    if (tmin == ty) return ny;
    return nz;
}

uint axis_mirror_mask;

#define FIRST_RAYMARCH_SUBTREE_FUNC f0
#define LAST_RAYMARCH_SUBTREE_FUNC f999

RayResult LAST_RAYMARCH_SUBTREE_FUNC(float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, Node n) {
    return NULL_RESULT;
}

#define DEF_RAYMARCH_SUBTREE(LEVEL, FUNC, NEXT_FUNC)                                                \
    RayResult FUNC(float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, Node n) {      \
        if (tx1 < 0.0 || ty1 < 0.0 || tz1 < 0.0 || n.value == 0) {                                  \
            return NULL_RESULT;                                                                     \
        }                                                                                           \
                                                                                                    \
        if ((LEVEL) == ubo.automaton.grid_depth - 1) {                                              \
            RayResult res;                                                                          \
            res.node = n;                                                                           \
            res.dist = max(tx0, ty0, tz0);                                                          \
                                                                                                    \
            if (res.dist == tx0) res.normal = vec3(1, 0, 0);                                        \
            else if (res.dist == ty0) res.normal = vec3(0, 1, 0);                                   \
            else if (res.dist == tz0) res.normal = vec3(0, 0, 1);                                   \
                                                                                                    \
            return res;                                                                             \
        }                                                                                           \
                                                                                                    \
        const float txm = 0.5 * (tx0 + tx1);                                                        \
        const float tym = 0.5 * (ty0 + ty1);                                                        \
        const float tzm = 0.5 * (tz0 + tz1);                                                        \
                                                                                                    \
        uint curr_node = first_node(tx0, ty0, tz0, txm, tym, tzm);                                  \
                                                                                                    \
        do {                                                                                        \
            float px0 = tx0;                                                                        \
            float py0 = ty0;                                                                        \
            float pz0 = tz0;                                                                        \
            float px1 = tx1;                                                                        \
            float py1 = ty1;                                                                        \
            float pz1 = tz1;                                                                        \
            Node child;                                                                             \
                                                                                                    \
            switch (curr_node) {                                                                    \
            case 0:                                                                                 \
                px1 = txm;                                                                          \
                py1 = tym;                                                                          \
                pz1 = tzm;                                                                          \
                child = get_child(n, axis_mirror_mask, (LEVEL));                                    \
                curr_node = new_node(txm, 4u, tym, 2u, tzm, 1u);                                    \
                break;                                                                              \
            case 1:                                                                                 \
                px1 = txm;                                                                          \
                py1 = tym;                                                                          \
                pz0 = tzm;                                                                          \
                child = get_child(n, 1u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = new_node(txm, 5u, tym, 3u, tz1, 8u);                                    \
                break;                                                                              \
            case 2:                                                                                 \
                px1 = txm;                                                                          \
                py0 = tym;                                                                          \
                pz1 = tzm;                                                                          \
                child = get_child(n, 2u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = new_node(txm, 6u, ty1, 8u, tzm, 3u);                                    \
                break;                                                                              \
            case 3:                                                                                 \
                px1 = txm;                                                                          \
                py0 = tym;                                                                          \
                pz0 = tzm;                                                                          \
                child = get_child(n, 3u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = new_node(txm, 7u, ty1, 8u, tz1, 8u);                                    \
                break;                                                                              \
            case 4:                                                                                 \
                px0 = txm;                                                                          \
                py1 = tym;                                                                          \
                pz1 = tzm;                                                                          \
                child = get_child(n, 4u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = new_node(tx1, 8u, tym, 6u, tzm, 5u);                                    \
                break;                                                                              \
            case 5:                                                                                 \
                px0 = txm;                                                                          \
                py1 = tym;                                                                          \
                pz0 = tzm;                                                                          \
                child = get_child(n, 5u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = new_node(tx1, 8u, tym, 7u, tz1, 8u);                                    \
                break;                                                                              \
            case 6:                                                                                 \
                px0 = txm;                                                                          \
                py0 = tym;                                                                          \
                pz1 = tzm;                                                                          \
                child = get_child(n, 6u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = new_node(tx1, 8u, ty1, 8u, tzm, 7u);                                    \
                break;                                                                              \
            case 7:                                                                                 \
                px0 = txm;                                                                          \
                py0 = tym;                                                                          \
                pz0 = tzm;                                                                          \
                child = get_child(n, 7u ^ axis_mirror_mask, (LEVEL));                               \
                curr_node = 8;                                                                      \
                break;                                                                              \
            default:                                                                                \
                break;                                                                              \
            }                                                                                       \
                                                                                                    \
            const RayResult res = NEXT_FUNC(px0, py0, pz0, px1, py1, pz1, child);                   \
            if (!is_null_result(res)) return res;                                                   \
                                                                                                    \
        } while (curr_node != 8);                                                                   \
                                                                                                    \
        return NULL_RESULT;                                                                         \
    }

DEF_RAYMARCH_SUBTREE(11u, f11, LAST_RAYMARCH_SUBTREE_FUNC)
DEF_RAYMARCH_SUBTREE(10u, f10, f11)
DEF_RAYMARCH_SUBTREE(9u, f9, f10)
DEF_RAYMARCH_SUBTREE(8u, f8, f9)
DEF_RAYMARCH_SUBTREE(7u, f7, f8)
DEF_RAYMARCH_SUBTREE(6u, f6, f7)
DEF_RAYMARCH_SUBTREE(5u, f5, f6)
DEF_RAYMARCH_SUBTREE(4u, f4, f5)
DEF_RAYMARCH_SUBTREE(3u, f3, f4)
DEF_RAYMARCH_SUBTREE(2u, f2, f3)
DEF_RAYMARCH_SUBTREE(1u, f1, f2)
DEF_RAYMARCH_SUBTREE(0u, FIRST_RAYMARCH_SUBTREE_FUNC, f1)

RayResult ray_parameter(Octree oct, Ray r) {
    vec3 normal_inverter = vec3(1);
    axis_mirror_mask = 0;

    if (r.direction.x < 0.0) {
        r.origin.x = 2 * oct.min.x + oct.size - r.origin.x;
        r.direction.x = -r.direction.x;
        axis_mirror_mask |= 4u;
        normal_inverter.x = -1;
    }

    if (r.direction.y < 0.0) {
        r.origin.y = 2 * oct.min.y + oct.size - r.origin.y;
        r.direction.y = -r.direction.y;
        axis_mirror_mask |= 2u;
        normal_inverter.y = -1;
    }

    if (r.direction.z < 0.0) {
        r.origin.z = 2 * oct.min.z + oct.size - r.origin.z;
        r.direction.z = -r.direction.z;
        axis_mirror_mask |= 1u;
        normal_inverter.z = -1;
    }

    const float tx0 = (oct.min.x - r.origin.x) / r.direction.x;
    const float tx1 = (oct.max.x - r.origin.x) / r.direction.x;
    const float ty0 = (oct.min.y - r.origin.y) / r.direction.y;
    const float ty1 = (oct.max.y - r.origin.y) / r.direction.y;
    const float tz0 = (oct.min.z - r.origin.z) / r.direction.z;
    const float tz1 = (oct.max.z - r.origin.z) / r.direction.z;

    if (max(tx0, ty0, tz0) < min(tx1, ty1, tz1)) {
        RayResult res = FIRST_RAYMARCH_SUBTREE_FUNC(tx0, ty0, tz0, tx1, ty1, tz1, oct.root);
        res.normal *= normal_inverter;
        return res;
    } else {
        return NULL_RESULT;
    }
}

bool check_neighbor(uvec3 checked, uvec3 delta) {
    const uvec3 neighbor_coords = checked + delta;
    const uint8_t value = read_value(morton3(neighbor_coords), ubo.automaton.grid_depth - 1);
    return value != 0;
}

#define COORD_RGB 0u
#define STATE_GRADIENT 1u
#define DISTANCE_GRADIENT 2u
#define SOLID_COLOR 3u

#define DEBUG_COLOR vec3(1, 0, 1)

vec3 get_node_color(Node n) {
    vec3 result = DEBUG_COLOR;

    if (ubo.coloring.preset_id == COORD_RGB) {
        const vec3 hit_voxel_coords = unmorton3(n.index);
        result = hit_voxel_coords / get_grid_width(ubo.automaton.grid_depth);

    } else if (ubo.coloring.preset_id == STATE_GRADIENT) {
        result = mix(
            ubo.coloring.cell_color1,
            ubo.coloring.cell_color2,
            float(n.value - uint8_t(1)) / (ubo.automaton.state_count - 1)
        );
    } else if (ubo.coloring.preset_id == DISTANCE_GRADIENT) {
        const vec3 hit_voxel_coords = unmorton3(n.index);
        const vec3 coords_centered = hit_voxel_coords - vec3(get_grid_width(ubo.automaton.grid_depth)) * 0.5;
        const float distance_from_center = length(coords_centered);
        const float max_distance = float(get_grid_width(ubo.automaton.grid_depth)) * 0.5 * sqrt(3.0);

        result = mix(
            ubo.coloring.cell_color1,
            ubo.coloring.cell_color2,
            distance_from_center / max_distance
        );
    } else if (ubo.coloring.preset_id == SOLID_COLOR) {
        result = ubo.coloring.cell_color1;
    }

    return result;
}

uint get_neighbor_count(uvec3 coords) {
    uint neighbor_count = 0;

    neighbor_count += uint(check_neighbor(coords, uvec3(1, 0, 0)));
    neighbor_count += uint(check_neighbor(coords, uvec3(-1, 0, 0)));
    neighbor_count += uint(check_neighbor(coords, uvec3(0, 1, 0)));
    neighbor_count += uint(check_neighbor(coords, uvec3(0, -1, 0)));
    neighbor_count += uint(check_neighbor(coords, uvec3(0, 0, 1)));
    neighbor_count += uint(check_neighbor(coords, uvec3(0, 0, -1)));

    return neighbor_count;
}

void main() {
    Node root_node;
    root_node.index = 0u;
    root_node.value = read_value(0u, 0u);

    Octree oct;
    oct.size = 16;
    oct.min = -0.5 * vec3(oct.size);
    oct.max = oct.min + vec3(oct.size);
    oct.root = root_node;

    vec2 frag_coord_normalized = 2.0 * gl_FragCoord.xy / vec2(ubo.window.width, ubo.window.height) - 1.0;
    frag_coord_normalized.y *= -1;

    const vec4 v = ubo.matrices.inverse_vp * vec4(frag_coord_normalized, 0, 1);
    const vec3 ray_pos = vec3(v) / v.w;

    Ray r;
    r.origin = ubo.misc.camera_pos;
    r.direction = normalize(ray_pos - ubo.misc.camera_pos);

    const RayResult ray_result = ray_parameter(oct, r);

    if (is_null_result(ray_result)) {
        outColor = vec4(ubo.coloring.background_color, 1.0);
        return;
    }

    outColor = vec4(get_node_color(ray_result.node), 1);

    // undo stupid ass gamma correction
    outColor = pow(outColor, vec4(2.2));

    // apply fake lighting
    const vec3 LIGHT_DIRECTION = normalize(vec3(1, 1.5, 2));
    const float d = max(dot(ray_result.normal, LIGHT_DIRECTION), dot(ray_result.normal, vec3(-1) * LIGHT_DIRECTION));
    outColor += vec4(vec3(d / 3), 1);

    // apply fake fog
    const float MIN_FOG = 0;
    const float MAX_FOG = 0.8;
    const float FOG_EXP = 1;
    outColor *= pow(1 - clamp(ray_result.dist / ubo.misc.fog_dist, MIN_FOG, MAX_FOG), FOG_EXP);

    // if desired, apply neighbor shading
    if (ubo.coloring.do_neighbor_shading != 0u) {
        const uvec3 hit_voxel_coords = unmorton3(ray_result.node.index);
        const uint neighbor_count = get_neighbor_count(hit_voxel_coords);
        outColor *= vec4(1.0 - float(neighbor_count) / 32);
    }
}
